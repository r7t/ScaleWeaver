#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using cd = std::complex<double>;
static constexpr double PI = 3.1415926535897932384626433832795;
static constexpr double EPS = 1e-15;

struct Parameters {
    double sigma = 1.0;
    double q = 0.8;
    double base_freq = 100.0;
    double max_freq = 4000.0;
    double resolution = 1.0;
};

struct Region { int start, end; };
struct NoteData { std::vector<double> ordinary, peak_max; };

static int next_pow2(int x) {
    int n = 1;
    while (n < x) n <<= 1;
    return n;
}

static void fft(std::vector<cd>& a, bool inverse) {
    const int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double angle = 2.0 * PI / len * (inverse ? -1.0 : 1.0);
        const cd step(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                const cd u = a[i + j];
                const cd v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= step;
            }
        }
    }
    if (inverse) for (auto& x : a) x /= static_cast<double>(n);
}

static std::uint64_t choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    std::uint64_t value = 1;
    for (int i = 1; i <= k; ++i) value = value * static_cast<std::uint64_t>(n - k + i) / i;
    return value;
}

static std::uint64_t colex_rank(const std::vector<int>& indices) {
    std::uint64_t rank = 0;
    for (std::size_t i = 0; i < indices.size(); ++i)
        rank += choose_u64(indices[i] + static_cast<int>(i), static_cast<int>(i) + 1);
    return rank;
}

static std::vector<int> parse_steps(const std::string& text) {
    std::vector<int> out;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) throw std::runtime_error("empty pitch in step list");
        out.push_back(std::stoi(token));
    }
    if (out.empty() || !std::is_sorted(out.begin(), out.end()) ||
        std::adjacent_find(out.begin(), out.end()) != out.end())
        throw std::runtime_error("step lists must be nonempty, sorted and unique");
    return out;
}

static NoteData make_note(int step, int edo, const Parameters& p) {
    const int n = static_cast<int>(std::ceil(p.max_freq / p.resolution)) + 1;
    NoteData note{std::vector<double>(n, 0.0), std::vector<double>(n, 0.0)};
    const double f0 = p.base_freq * std::exp2(static_cast<double>(step) / edo);
    for (int harmonic = 1;; ++harmonic) {
        const double frequency = f0 * harmonic;
        if (frequency >= p.max_freq) break;
        const double amplitude = std::pow(p.q, (frequency - f0) / 100.0);
        if (amplitude < 0.001) break;
        const int lo = std::max(0, static_cast<int>((frequency - 5.0 * p.sigma) / p.resolution));
        const int hi = std::min(n - 1, static_cast<int>((frequency + 5.0 * p.sigma) / p.resolution));
        for (int k = lo; k <= hi; ++k) {
            const double d = k * p.resolution - frequency;
            const double value = amplitude * std::exp(-(d * d) / (2.0 * p.sigma * p.sigma));
            note.ordinary[k] += value;
            note.peak_max[k] = std::max(note.peak_max[k], value);
        }
    }
    return note;
}

struct Scratch {
    int spectrum_n, convolution_n, fft_n;
    std::vector<double> ordinary, peak_max, cancelled, total, pair_max, pair_conv;
    std::vector<cd> fourier;
    std::vector<Region> regions;

    Scratch(int sn, int fn)
        : spectrum_n(sn), convolution_n(2 * sn - 1), fft_n(fn),
          ordinary(sn), peak_max(sn), cancelled(sn), total(convolution_n),
          pair_max(convolution_n), pair_conv(2 * sn - 1), fourier(fn) {}
};

static std::pair<double, double> score_combo(
    const std::vector<int>& combo, const std::vector<NoteData>& notes, Scratch& s) {
    std::fill(s.ordinary.begin(), s.ordinary.end(), 0.0);
    std::fill(s.peak_max.begin(), s.peak_max.end(), 0.0);
    const int root = combo.front();
    for (int step : combo) {
        const auto& note = notes.at(step - root);
        for (int k = 0; k < s.spectrum_n; ++k) {
            s.ordinary[k] += note.ordinary[k];
            s.peak_max[k] = std::max(s.peak_max[k], note.peak_max[k]);
        }
    }

    long double ordinary_energy = 0.0L, cancelled_energy = 0.0L;
    s.regions.clear();
    bool inside = false;
    int region_start = 0;
    for (int k = 0; k < s.spectrum_n; ++k) {
        const double value = std::max(0.0, 2.0 * s.peak_max[k] - s.ordinary[k]);
        s.cancelled[k] = value;
        ordinary_energy += static_cast<long double>(s.ordinary[k]) * s.ordinary[k];
        cancelled_energy += static_cast<long double>(value) * value;
        if (value > EPS && !inside) { inside = true; region_start = k; }
        if (inside && (value <= EPS || k == s.spectrum_n - 1)) {
            const int end = (value <= EPS ? k : k + 1);
            s.regions.push_back({region_start, end});
            inside = false;
        }
    }
    const double o1 = ordinary_energy > 0.0L
        ? 1.0 - static_cast<double>(cancelled_energy / ordinary_energy) : 0.0;

    std::fill(s.fourier.begin(), s.fourier.end(), cd(0.0, 0.0));
    for (int k = 0; k < s.spectrum_n; ++k) s.fourier[k] = cd(s.cancelled[k], 0.0);
    fft(s.fourier, false);
    for (auto& value : s.fourier) value *= value;
    fft(s.fourier, true);
    long double normal_convolution_energy = 0.0L;
    for (int k = 0; k < s.convolution_n; ++k) {
        const double value = std::max(0.0, s.fourier[k].real());
        normal_convolution_energy += static_cast<long double>(value) * value;
    }

    std::fill(s.total.begin(), s.total.end(), 0.0);
    std::fill(s.pair_max.begin(), s.pair_max.end(), 0.0);
    for (std::size_t i = 0; i < s.regions.size(); ++i) {
        const Region& a = s.regions[i];
        for (std::size_t j = i; j < s.regions.size(); ++j) {
            const Region& b = s.regions[j];
            const int na = a.end - a.start, nb = b.end - b.start;
            const int length = na + nb - 1;
            std::fill(s.pair_conv.begin(), s.pair_conv.begin() + length, 0.0);
            const double multiplier = (i == j ? 1.0 : 2.0);
            for (int x = 0; x < na; ++x) {
                const double av = multiplier * s.cancelled[a.start + x];
                for (int y = 0; y < nb; ++y)
                    s.pair_conv[x + y] += av * s.cancelled[b.start + y];
            }
            const int start = a.start + b.start;
            for (int t = 0; t < length; ++t) {
                const double value = s.pair_conv[t];
                s.total[start + t] += value;
                s.pair_max[start + t] = std::max(s.pair_max[start + t], value);
            }
        }
    }
    long double phase_cancelled_energy = 0.0L;
    for (int k = 0; k < s.convolution_n; ++k) {
        const double value = std::max(0.0, 2.0 * s.pair_max[k] - s.total[k]);
        phase_cancelled_energy += static_cast<long double>(value) * value;
    }
    const double o2 = normal_convolution_energy > 0.0L
        ? 1.0 - static_cast<double>(phase_cancelled_energy / normal_convolution_energy) : 0.0;
    return {std::clamp(o1, 0.0, 1.0), std::clamp(o2, 0.0, 1.0)};
}

static void combinations_rec(int n, int card, int pos, int minimum,
                             std::vector<int>& current, std::vector<std::vector<int>>& out) {
    if (pos == card) { out.push_back(current); return; }
    for (int value = minimum; value < n; ++value) {
        current[pos] = value;
        combinations_rec(n, card, pos + 1, value, current, out);
    }
}

static void build_table(const std::vector<int>& table_steps, int card,
                        const std::vector<NoteData>& notes, const Parameters& p,
                        int workers, int progress_every,
                        std::ofstream& o1_file, std::ofstream& o2_file) {
    std::vector<std::vector<int>> local_combos;
    std::vector<int> current(card);
    combinations_rec(static_cast<int>(table_steps.size()), card, 0, 0, current, local_combos);
    const std::uint64_t count = choose_u64(static_cast<int>(table_steps.size()) + card - 1, card);
    if (local_combos.size() != count) throw std::runtime_error("combination count mismatch");
    std::vector<std::vector<int>> combos(count, std::vector<int>(card));
    for (const auto& combo : local_combos) combos[colex_rank(combo)] = combo;
    local_combos.clear(); local_combos.shrink_to_fit();
    for (auto& combo : combos) for (int& index : combo) index = table_steps[index];

    std::vector<double> o1(count), o2(count);
    const int spectrum_n = static_cast<int>(std::ceil(p.max_freq / p.resolution)) + 1;
    const int fft_n = next_pow2(2 * spectrum_n - 1);
    std::atomic<std::uint64_t> done{0};
    const auto started = std::chrono::steady_clock::now();
#ifdef _OPENMP
    omp_set_num_threads(workers);
#pragma omp parallel
#endif
    {
        Scratch scratch(spectrum_n, fft_n);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
        for (std::int64_t rank = 0; rank < static_cast<std::int64_t>(count); ++rank) {
            const auto value = score_combo(combos[rank], notes, scratch);
            o1[rank] = value.first;
            o2[rank] = value.second;
            const std::uint64_t completed = done.fetch_add(1, std::memory_order_relaxed) + 1;
            if (progress_every > 0 && (completed % progress_every == 0 || completed == count)) {
#ifdef _OPENMP
#pragma omp critical(overlap_progress)
#endif
                {
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count();
                    std::cerr << "O1/O2 k=" << card << ": " << completed << "/" << count
                              << " (" << std::fixed << std::setprecision(0)
                              << completed / std::max(1e-9, seconds) << " chords/s)\n";
                }
            }
        }
    }
    o1_file.write(reinterpret_cast<const char*>(o1.data()), static_cast<std::streamsize>(count * sizeof(double)));
    o2_file.write(reinterpret_cast<const char*>(o2.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if (!o1_file || !o2_file) throw std::runtime_error("failed to write raw overlap table");
}

int main(int argc, char** argv) {
    try {
        std::unordered_map<std::string, std::string> args;
        for (int i = 1; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];
        const int edo = std::stoi(args.at("--edo"));
        const int workers = std::stoi(args.at("--workers"));
        const int progress = std::stoi(args.at("--progress-every"));
        Parameters p;
        p.sigma = std::stod(args.at("--sigma"));
        p.q = std::stod(args.at("--q"));
        p.base_freq = std::stod(args.at("--base-freq"));
        p.max_freq = std::stod(args.at("--max-freq"));
        p.resolution = std::stod(args.at("--resolution"));
        const std::vector<int> wide = parse_steps(args.at("--wide"));
        const std::vector<int> inner = parse_steps(args.at("--inner"));
        for (int step : inner)
            if (!std::binary_search(wide.begin(), wide.end(), step))
                throw std::runtime_error("inner domain is not a subset of wide domain");

        std::vector<NoteData> notes;
        const int maximum_interval = wide.back() - wide.front();
        notes.reserve(maximum_interval + 1);
        for (int step = 0; step <= maximum_interval; ++step)
            notes.push_back(make_note(step, edo, p));
        std::ofstream o1(args.at("--o1"), std::ios::binary);
        std::ofstream o2(args.at("--o2"), std::ios::binary);
        if (!o1 || !o2) throw std::runtime_error("cannot open overlap output files");
        for (int card : {2, 3, 4}) build_table(wide, card, notes, p, workers, progress, o1, o2);
        build_table(inner, 5, notes, p, workers, progress, o1, o2);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "overlap_precompute: " << e.what() << "\n";
        return 1;
    }
}
