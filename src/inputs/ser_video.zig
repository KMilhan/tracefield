const std = @import("std");

pub const SerHeader = struct {
    width: u32,
    height: u32,
    pixel_depth: u32,
    frame_count: u32,
};

pub const SerInfo = struct {
    header: SerHeader,
    frame_count: u32,
};

pub const FrameRange = struct {
    start: u32,
    end: u32,
};

pub const SerMetrics = struct {
    frame_count: u32,
    centroid_rms: f64,
    high_freq_rms: f64,
    peak_excursions: u32,
    centroid_energy: f64,
    peak_burst_rate: f64,
    fwhm_burst_rate: f64,
    settle_peak_fraction: f64,
    centroid_covariance: f64,
};

pub const ParseError = std.mem.Allocator.Error ||
    std.fs.File.OpenError ||
    std.fs.File.ReadError ||
    std.fs.File.SeekError ||
    std.fs.File.GetEndPosError ||
    error{EndOfStream} ||
    error{
        InvalidHeader,
        UnsupportedDepth,
        InvalidFrameCount,
    };

const header_size = 178;

pub fn readInfo(path: []const u8) ParseError!SerInfo {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const file_size = try file.getEndPos();
    const reader = file.deprecatedReader();
    const header = try readHeader(reader);

    const bytes_per_pixel: usize = @intCast(header.pixel_depth / 8);
    const frame_bytes = @as(usize, header.width) * @as(usize, header.height) * bytes_per_pixel;
    var frame_count = header.frame_count;
    if (frame_bytes > 0 and file_size > header_size) {
        const derived = @as(u32, @intCast((file_size - header_size) / frame_bytes));
        if (derived > 0) frame_count = derived;
    }
    if (frame_count == 0) return error.InvalidFrameCount;

    return .{ .header = header, .frame_count = frame_count };
}

pub fn readHeader(reader: anytype) ParseError!SerHeader {
    var header_buf: [header_size]u8 = undefined;
    try reader.readNoEof(&header_buf);

    const file_id = header_buf[0..14];
    if (!std.mem.startsWith(u8, file_id, "INDI-RECORDER") and
        !std.mem.startsWith(u8, file_id, "LUCAM-RECORDER") and
        !std.mem.startsWith(u8, file_id, "SER"))
    {
        return error.InvalidHeader;
    }

    const base = 14;
    const width = std.mem.readInt(u32, header_buf[base + 3 * 4 .. base + 4 * 4], .little);
    const height = std.mem.readInt(u32, header_buf[base + 4 * 4 .. base + 5 * 4], .little);
    const pixel_depth = std.mem.readInt(u32, header_buf[base + 5 * 4 .. base + 6 * 4], .little);
    var frame_count = std.mem.readInt(u32, header_buf[base + 6 * 4 .. base + 7 * 4], .little);
    if (frame_count == 0) frame_count = 1;

    if (pixel_depth != 8 and pixel_depth != 16) return error.UnsupportedDepth;

    return .{
        .width = width,
        .height = height,
        .pixel_depth = pixel_depth,
        .frame_count = frame_count,
    };
}

pub fn analyzeFile(path: []const u8, allocator: std.mem.Allocator) ParseError!SerMetrics {
    return analyzeFileRange(path, allocator, null);
}

pub fn analyzeFileRange(path: []const u8, allocator: std.mem.Allocator, range: ?FrameRange) ParseError!SerMetrics {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    var reader = file.deprecatedReader();
    const header = try readHeader(reader);

    const bytes_per_pixel: usize = @intCast(header.pixel_depth / 8);
    const frame_bytes = @as(usize, header.width) * @as(usize, header.height) * bytes_per_pixel;
    var frame_count = header.frame_count;
    if (frame_bytes > 0 and file_size > header_size) {
        const derived = @as(u32, @intCast((file_size - header_size) / frame_bytes));
        if (derived > 0) frame_count = derived;
    }
    if (frame_count == 0) return error.InvalidFrameCount;

    var start_frame: u32 = 0;
    var end_frame: u32 = frame_count;
    if (range) |r| {
        start_frame = @min(r.start, frame_count);
        end_frame = @min(r.end, frame_count);
        if (end_frame < start_frame) end_frame = start_frame;
    }
    const range_count = end_frame - start_frame;
    if (range_count == 0) return error.InvalidFrameCount;

    try file.seekTo(@as(u64, header_size) + @as(u64, start_frame) * @as(u64, frame_bytes));
    reader = file.deprecatedReader();

    var centroids_x = try allocator.alloc(f64, range_count);
    var centroids_y = try allocator.alloc(f64, range_count);
    defer allocator.free(centroids_x);
    defer allocator.free(centroids_y);

    const frame_buf = try allocator.alloc(u8, frame_bytes);
    defer allocator.free(frame_buf);

    var i: usize = 0;
    while (i < range_count) : (i += 1) {
        try reader.readNoEof(frame_buf);
        var sum: f64 = 0.0;
        var sum_x: f64 = 0.0;
        var sum_y: f64 = 0.0;

        var y: usize = 0;
        var offset: usize = 0;
        while (y < header.height) : (y += 1) {
            var x: usize = 0;
            while (x < header.width) : (x += 1) {
                var value: f64 = 0.0;
                if (header.pixel_depth == 8) {
                    value = @as(f64, @floatFromInt(frame_buf[offset]));
                    offset += 1;
                } else {
                    var two: [2]u8 = .{ frame_buf[offset], frame_buf[offset + 1] };
                    const raw = std.mem.readInt(u16, &two, .little);
                    value = @as(f64, @floatFromInt(raw));
                    offset += 2;
                }
                sum += value;
                sum_x += value * @as(f64, @floatFromInt(x));
                sum_y += value * @as(f64, @floatFromInt(y));
            }
        }

        if (sum > 0.0) {
            centroids_x[i] = sum_x / sum;
            centroids_y[i] = sum_y / sum;
        } else {
            centroids_x[i] = 0.0;
            centroids_y[i] = 0.0;
        }
    }

    var mean_x: f64 = 0.0;
    var mean_y: f64 = 0.0;
    for (centroids_x) |v| mean_x += v;
    for (centroids_y) |v| mean_y += v;
    mean_x /= @as(f64, @floatFromInt(range_count));
    mean_y /= @as(f64, @floatFromInt(range_count));

    var rms_sum: f64 = 0.0;
    var dist_sum: f64 = 0.0;
    var dist_sq_sum: f64 = 0.0;
    var hf_sum: f64 = 0.0;
    var cov_sum: f64 = 0.0;

    i = 0;
    while (i < range_count) : (i += 1) {
        const dx = centroids_x[i] - mean_x;
        const dy = centroids_y[i] - mean_y;
        const dist = @sqrt(dx * dx + dy * dy);
        rms_sum += dist * dist;
        dist_sum += dist;
        dist_sq_sum += dist * dist;
        cov_sum += dx * dy;
        if (i > 0) {
            const ddx = centroids_x[i] - centroids_x[i - 1];
            const ddy = centroids_y[i] - centroids_y[i - 1];
            hf_sum += ddx * ddx + ddy * ddy;
        }
    }

    const mean_dist = dist_sum / @as(f64, @floatFromInt(range_count));
    const mean_dist_sq = dist_sq_sum / @as(f64, @floatFromInt(range_count));
    const var_dist = mean_dist_sq - mean_dist * mean_dist;
    const std_dist = if (var_dist > 0.0) @sqrt(var_dist) else 0.0;

    var peaks: u32 = 0;
    var settle_peaks: u32 = 0;
    const settle_frames: u32 = @max(@as(u32, 1), range_count / 10);
    i = 0;
    while (i < range_count) : (i += 1) {
        const dx = centroids_x[i] - mean_x;
        const dy = centroids_y[i] - mean_y;
        const dist = @sqrt(dx * dx + dy * dy);
        if (dist > mean_dist + 3.0 * std_dist) {
            peaks += 1;
            if (i < settle_frames) settle_peaks += 1;
        }
    }

    const centroid_rms = @sqrt(rms_sum / @as(f64, @floatFromInt(range_count)));
    const high_freq_rms = if (range_count > 1)
        @sqrt(hf_sum / @as(f64, @floatFromInt(range_count - 1)))
    else
        0.0;
    const peak_burst_rate = @as(f64, @floatFromInt(peaks)) / @as(f64, @floatFromInt(range_count));
    const fwhm_burst_rate = peak_burst_rate;
    const settle_peak_fraction = if (settle_frames > 0)
        @as(f64, @floatFromInt(settle_peaks)) / @as(f64, @floatFromInt(settle_frames))
    else
        0.0;
    const centroid_covariance = cov_sum / @as(f64, @floatFromInt(range_count));

    return .{
        .frame_count = @as(u32, @intCast(range_count)),
        .centroid_rms = centroid_rms,
        .high_freq_rms = high_freq_rms,
        .peak_excursions = peaks,
        .centroid_energy = mean_dist_sq,
        .peak_burst_rate = peak_burst_rate,
        .fwhm_burst_rate = fwhm_burst_rate,
        .settle_peak_fraction = settle_peak_fraction,
        .centroid_covariance = centroid_covariance,
    };
}

test "parse ser header" {
    var file = try std.fs.cwd().openFile("testdata/secondary/guide2.ser", .{});
    defer file.close();
    const header = try readHeader(file.reader());
    try std.testing.expectEqual(@as(u32, 640), header.width);
    try std.testing.expectEqual(@as(u32, 480), header.height);
    try std.testing.expectEqual(@as(u32, 16), header.pixel_depth);
    try std.testing.expectEqual(@as(u32, 50), header.frame_count);
}

test "analyze ser range" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const metrics = try analyzeFileRange("testdata/secondary/guide2.ser", allocator, .{ .start = 0, .end = 10 });
    try std.testing.expectEqual(@as(u32, 10), metrics.frame_count);
}
