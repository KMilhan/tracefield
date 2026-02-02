const std = @import("std");
const inputs = @import("inputs.zig");

pub const MainMetrics = struct {
    fwhm: f64,
    eccentricity: f64,
    star_count: u64,
    snr: f64,
};

pub const Phd2Metrics = struct {
    rms_error: f64,
    peak_to_peak: f64,
    settle_rms: f64,
    settle_fraction: f64,
};

pub const SecondaryMetrics = struct {
    centroid_rms: f64,
    high_freq_rms: f64,
    peak_excursions: u32,
    centroid_energy: f64,
    peak_burst_rate: f64,
    fwhm_burst_rate: f64,
    settle_peak_fraction: f64,
    centroid_covariance: f64,
};

fn histogramMedian(hist: *const [65536]u32, pixel_count: u64) u16 {
    if (pixel_count == 0) return 0;
    const target = (pixel_count + 1) / 2;
    var acc: u64 = 0;
    var i: usize = 0;
    while (i < hist.len) : (i += 1) {
        acc += hist[i];
        if (acc >= target) return @as(u16, @intCast(i));
    }
    return 0;
}

pub fn computeMainMetrics(image: inputs.fits_image.FitsImage) MainMetrics {
    const width = image.width;
    const height = image.height;
    const pixel_count = width * height;

    const Candidate = struct {
        x: usize,
        y: usize,
        v: f64,
    };

    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    var max_val: f64 = 0.0;
    var min_val: f64 = std.math.inf(f64);
    var hist: [65536]u32 = [_]u32{0} ** 65536;

    for (image.pixels) |px| {
        const val = @as(f64, @floatFromInt(px));
        sum += val;
        sum_sq += val * val;
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
        hist[px] += 1;
    }

    const mean = sum / @as(f64, @floatFromInt(pixel_count));
    const variance = sum_sq / @as(f64, @floatFromInt(pixel_count)) - mean * mean;
    const stddev = if (variance > 0.0) @sqrt(variance) else 0.0;

    const median_val: u16 = histogramMedian(&hist, pixel_count);

    var dev_hist: [65536]u32 = [_]u32{0} ** 65536;
    for (image.pixels) |px| {
        const diff = if (px > median_val) px - median_val else median_val - px;
        dev_hist[diff] += 1;
    }

    const mad_val: u16 = histogramMedian(&dev_hist, pixel_count);

    const sigma = 1.4826 * @as(f64, @floatFromInt(mad_val));
    const max_u16 = @as(u16, @intFromFloat(@min(max_val, 65535.0)));
    const p99_5: u16 = blk: {
        const target = (pixel_count * 995) / 1000;
        var acc: u64 = 0;
        var i: usize = 0;
        while (i < hist.len) : (i += 1) {
            acc += hist[i];
            if (acc >= target) break :blk @as(u16, @intCast(i));
        }
        break :blk max_u16;
    };

    const p99: u16 = blk: {
        const target = (pixel_count * 990) / 1000;
        var acc: u64 = 0;
        var i: usize = 0;
        while (i < hist.len) : (i += 1) {
            acc += hist[i];
            if (acc >= target) break :blk @as(u16, @intCast(i));
        }
        break :blk max_u16;
    };

    const p98: u16 = blk: {
        const target = (pixel_count * 980) / 1000;
        var acc: u64 = 0;
        var i: usize = 0;
        while (i < hist.len) : (i += 1) {
            acc += hist[i];
            if (acc >= target) break :blk @as(u16, @intCast(i));
        }
        break :blk max_u16;
    };

    const threshold_levels = [_]f64{
        @max(@as(f64, @floatFromInt(median_val)) + 2.0 * sigma, @as(f64, @floatFromInt(p99_5))),
        @max(@as(f64, @floatFromInt(median_val)) + 1.5 * sigma, @as(f64, @floatFromInt(p99))),
        @max(@as(f64, @floatFromInt(median_val)) + 1.0 * sigma, @as(f64, @floatFromInt(p98))),
    };
    var threshold = threshold_levels[0];

    const max_candidates = 4096;
    var candidates: [max_candidates]Candidate = undefined;
    var candidate_count: usize = 0;
    var total_peaks: u64 = 0;

    var pass: usize = 0;
    while (pass < threshold_levels.len) : (pass += 1) {
        candidate_count = 0;
        total_peaks = 0;
        threshold = threshold_levels[pass];
        var y: usize = 1;
        while (y + 1 < height) : (y += 1) {
            var x: usize = 1;
            while (x + 1 < width) : (x += 1) {
                const idx = y * width + x;
                const v = @as(f64, @floatFromInt(image.pixels[idx]));
                if (v <= threshold) continue;

                const up = @as(f64, @floatFromInt(image.pixels[idx - width]));
                const down = @as(f64, @floatFromInt(image.pixels[idx + width]));
                const left = @as(f64, @floatFromInt(image.pixels[idx - 1]));
                const right = @as(f64, @floatFromInt(image.pixels[idx + 1]));
                const ul = @as(f64, @floatFromInt(image.pixels[idx - width - 1]));
                const ur = @as(f64, @floatFromInt(image.pixels[idx - width + 1]));
                const dl = @as(f64, @floatFromInt(image.pixels[idx + width - 1]));
                const dr = @as(f64, @floatFromInt(image.pixels[idx + width + 1]));
                var strictly_greater = false;
                if (v < up or v < down or v < left or v < right or v < ul or v < ur or v < dl or v < dr) {
                    continue;
                }
                if (v > up or v > down or v > left or v > right or v > ul or v > ur or v > dl or v > dr) {
                    strictly_greater = true;
                }
                if (!strictly_greater) continue;

                total_peaks += 1;
                if (candidate_count < max_candidates) {
                    candidates[candidate_count] = .{ .x = x, .y = y, .v = v };
                    candidate_count += 1;
                } else {
                    var min_i: usize = 0;
                    var min_v: f64 = candidates[0].v;
                    var i: usize = 1;
                    while (i < max_candidates) : (i += 1) {
                        if (candidates[i].v < min_v) {
                            min_v = candidates[i].v;
                            min_i = i;
                        }
                    }
                    if (v > min_v) {
                        candidates[min_i] = .{ .x = x, .y = y, .v = v };
                    }
                }
            }
        }
        if (total_peaks > 0) break;
    }

    if (candidate_count > 1) {
        std.sort.block(Candidate, candidates[0..candidate_count], {}, struct {
            pub fn lessThan(_: void, a: Candidate, b: Candidate) bool {
                return a.v > b.v;
            }
        }.lessThan);
    }

    const max_stars = 200;
    var star_x: [max_stars]usize = undefined;
    var star_y: [max_stars]usize = undefined;
    var star_val: [max_stars]f64 = undefined;
    var kept: usize = 0;
    const min_dist2: i32 = 64;
    for (candidates[0..candidate_count]) |cand| {
        if (kept >= max_stars) break;
        var too_close = false;
        var i: usize = 0;
        while (i < kept) : (i += 1) {
            const dx = @as(i32, @intCast(cand.x)) - @as(i32, @intCast(star_x[i]));
            const dy = @as(i32, @intCast(cand.y)) - @as(i32, @intCast(star_y[i]));
            if (dx * dx + dy * dy <= min_dist2) {
                too_close = true;
                break;
            }
        }
        if (!too_close) {
            star_x[kept] = cand.x;
            star_y[kept] = cand.y;
            star_val[kept] = cand.v;
            kept += 1;
        }
    }

    var fwhm_sum: f64 = 0.0;
    var ecc_sum: f64 = 0.0;
    var used: u64 = 0;
    const radius: i32 = 4;
    const background = @as(f64, @floatFromInt(median_val));
    var i: usize = 0;
    while (i < kept) : (i += 1) {
        const sx = @as(i32, @intCast(star_x[i]));
        const sy = @as(i32, @intCast(star_y[i]));
        var sum_w: f64 = 0.0;
        var sum_x: f64 = 0.0;
        var sum_y: f64 = 0.0;
        var sum_xx: f64 = 0.0;
        var sum_yy: f64 = 0.0;

        var dy: i32 = -radius;
        while (dy <= radius) : (dy += 1) {
            const yy = sy + dy;
            if (yy < 0 or yy >= @as(i32, @intCast(height))) continue;
            var dx: i32 = -radius;
            while (dx <= radius) : (dx += 1) {
                const xx = sx + dx;
                if (xx < 0 or xx >= @as(i32, @intCast(width))) continue;
                const idx2 = @as(usize, @intCast(yy)) * width + @as(usize, @intCast(xx));
                const raw = @as(f64, @floatFromInt(image.pixels[idx2]));
                const w = @max(0.0, raw - background);
                if (w == 0.0) continue;
                const xf = @as(f64, @floatFromInt(xx));
                const yf = @as(f64, @floatFromInt(yy));
                sum_w += w;
                sum_x += w * xf;
                sum_y += w * yf;
                sum_xx += w * xf * xf;
                sum_yy += w * yf * yf;
            }
        }

        if (sum_w > 0.0) {
            const cx = sum_x / sum_w;
            const cy = sum_y / sum_w;
            const var_x = sum_xx / sum_w - cx * cx;
            const var_y = sum_yy / sum_w - cy * cy;
            const sigma_star = @sqrt(@max(0.0, (var_x + var_y) / 2.0));
            const fwhm = 2.355 * sigma_star;
            const max_var = @max(var_x, var_y);
            const min_var = @min(var_x, var_y);
            const ecc = if (max_var > 0.0 and min_var >= 0.0)
                @sqrt(1.0 - (min_var / max_var))
            else
                0.0;
            fwhm_sum += fwhm;
            ecc_sum += ecc;
            used += 1;
        }
    }

    const fwhm = if (used > 0) fwhm_sum / @as(f64, @floatFromInt(used)) else 0.0;
    const ecc = if (used > 0) ecc_sum / @as(f64, @floatFromInt(used)) else 0.0;
    const snr = if (stddev > 0.0) (max_val - mean) / stddev else 0.0;
    const star_count = kept;

    return .{
        .fwhm = fwhm,
        .eccentricity = ecc,
        .star_count = star_count,
        .snr = snr,
    };
}

pub fn computePhd2Metrics(samples: []const inputs.phd2_log.Sample, settle_window_s: f64) Phd2Metrics {
    if (samples.len == 0) {
        return .{
            .rms_error = 0.0,
            .peak_to_peak = 0.0,
            .settle_rms = 0.0,
            .settle_fraction = 0.0,
        };
    }

    var sum_sq: f64 = 0.0;
    var max_v: f64 = -std.math.inf(f64);
    var min_v: f64 = std.math.inf(f64);
    var settle_sum_sq: f64 = 0.0;
    var settle_count: u64 = 0;
    const t0 = samples[0].timestamp_ns;
    for (samples) |s| {
        const mag = @sqrt(s.ra_error_arcsec * s.ra_error_arcsec + s.dec_error_arcsec * s.dec_error_arcsec);
        sum_sq += mag * mag;
        if (mag > max_v) max_v = mag;
        if (mag < min_v) min_v = mag;

        const dt_s = @as(f64, @floatFromInt(s.timestamp_ns - t0)) / 1_000_000_000.0;
        if (dt_s <= settle_window_s) {
            settle_sum_sq += mag * mag;
            settle_count += 1;
        }
    }
    const rms = @sqrt(sum_sq / @as(f64, @floatFromInt(samples.len)));
    const p2p = if (min_v <= max_v) max_v - min_v else 0.0;
    const settle_rms = if (settle_count > 0)
        @sqrt(settle_sum_sq / @as(f64, @floatFromInt(settle_count)))
    else
        0.0;
    const settle_fraction = @as(f64, @floatFromInt(settle_count)) / @as(f64, @floatFromInt(samples.len));

    return .{
        .rms_error = rms,
        .peak_to_peak = p2p,
        .settle_rms = settle_rms,
        .settle_fraction = settle_fraction,
    };
}

pub fn computeSecondaryMetrics(metrics: inputs.ser_video.SerMetrics) SecondaryMetrics {
    return .{
        .centroid_rms = metrics.centroid_rms,
        .high_freq_rms = metrics.high_freq_rms,
        .peak_excursions = metrics.peak_excursions,
        .centroid_energy = metrics.centroid_energy,
        .peak_burst_rate = metrics.peak_burst_rate,
        .fwhm_burst_rate = metrics.fwhm_burst_rate,
        .settle_peak_fraction = metrics.settle_peak_fraction,
        .centroid_covariance = metrics.centroid_covariance,
    };
}

pub fn computeWeight(main: MainMetrics, phd2: Phd2Metrics, secondary: SecondaryMetrics) f64 {
    const main_score = if (main.star_count == 0) 0.25 else
        1.0 / (1.0 + (main.fwhm / 4.0) + (main.eccentricity * 2.0));
    const phd2_score = 1.0 / (1.0 + (phd2.rms_error / 1.0) + (phd2.peak_to_peak / 2.0) + (phd2.settle_rms / 1.0));
    const settle_penalty = 1.0 - @min(0.5, phd2.settle_fraction * 0.5);
    const sec_score = 1.0 / (1.0 + (secondary.centroid_rms / 2.0) + (secondary.high_freq_rms / 2.0) +
        (secondary.centroid_energy / 2.0) + (secondary.peak_burst_rate * 3.0));
    const sec_settle_penalty = 1.0 - @min(0.4, secondary.settle_peak_fraction * 0.4);

    return ((main_score + phd2_score + sec_score) / 3.0) * settle_penalty * sec_settle_penalty;
}

test "histogram median uses middle value for odd counts" {
    var hist: [65536]u32 = [_]u32{0} ** 65536;
    hist[0] = 1;
    hist[10] = 1;
    hist[20] = 1;
    const median = histogramMedian(&hist, 3);
    try std.testing.expectEqual(@as(u16, 10), median);
}
