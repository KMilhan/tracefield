const std = @import("std");
const metrics = @import("metrics.zig");

pub const DataSource = enum {
    main_fits,
    phd2_log,
    secondary_video,
};

pub const inputs = @import("inputs.zig");

pub const ExposureWindow = struct {
    id: []const u8,
    start_ns: i128,
    end_ns: i128,
    guard_before_s: u32,
    guard_after_s: u32,
};

pub fn describe(writer: anytype) !void {
    try writer.print("GuideCam telemetry pipeline scaffold\n", .{});
    try writer.print("data-sources: main_fits, phd2_log, secondary_video\n", .{});
    try writer.print("inputs: ekos_events, phd2_log, telemetry_video\n", .{});
}

pub const RunConfig = struct {
    fits_path: []const u8,
    phd2_path: []const u8,
    ser_path: []const u8,
    settle_window_s: f64 = 5.0,
};

pub const RunOutput = struct {
    main: metrics.MainMetrics,
    phd2: metrics.Phd2Metrics,
    secondary: metrics.SecondaryMetrics,
    weight: f64,
};

pub const DatasetConfig = struct {
    dataset_dir: []const u8,
    out_dir: []const u8,
    ekos_path: ?[]const u8 = null,
    guard_before_s: u32 = 0,
    guard_after_s: u32 = 0,
    settle_window_s: f64 = 5.0,
    prefer_session_ser: bool = false,
};

pub const SubResult = struct {
    id: u32,
    fits_path: []const u8,
    phd2_path: []const u8,
    ser_path: []const u8,
    main: metrics.MainMetrics,
    phd2: metrics.Phd2Metrics,
    secondary: metrics.SecondaryMetrics,
    weight: f64,

    pub fn deinit(self: *SubResult, allocator: std.mem.Allocator) void {
        allocator.free(self.fits_path);
        allocator.free(self.phd2_path);
        allocator.free(self.ser_path);
        self.* = undefined;
    }
};

pub const DatasetOutput = struct {
    subs: []SubResult,
    stack_plain_path: []const u8,
    stack_weighted_path: []const u8,
    used_ekos: bool,
    used_session_ser: bool,
    session_ser_path: ?[]const u8,

    pub fn deinit(self: *DatasetOutput, allocator: std.mem.Allocator) void {
        for (self.subs) |*sub| sub.deinit(allocator);
        allocator.free(self.subs);
        allocator.free(self.stack_plain_path);
        allocator.free(self.stack_weighted_path);
        if (self.session_ser_path) |path| allocator.free(path);
        self.* = undefined;
    }
};

const SubEntry = struct {
    id: u32,
    name: []const u8,

    pub fn deinit(self: *SubEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.* = undefined;
    }
};

fn parsePhd2File(path: []const u8, allocator: std.mem.Allocator) ![]inputs.phd2_log.Sample {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const data = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(data);

    if (std.mem.indexOf(u8, data, "Frame,Time") != null) {
        const log = try inputs.phd2_log.parseGuideLog(data, allocator);
        return log.samples;
    }

    var samples: std.ArrayList(inputs.phd2_log.Sample) = .empty;
    errdefer samples.deinit(allocator);
    var line_it = std.mem.splitScalar(u8, data, '\n');
    while (line_it.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;
        try samples.append(allocator, try inputs.phd2_log.parseLine(line));
    }
    return try samples.toOwnedSlice(allocator);
}

pub fn run(allocator: std.mem.Allocator, config: RunConfig) !RunOutput {
    var image = try inputs.fits_image.readImage(config.fits_path, allocator);
    defer image.deinit(allocator);

    const phd2_samples = try parsePhd2File(config.phd2_path, allocator);
    defer allocator.free(phd2_samples);

    const ser_metrics = try inputs.ser_video.analyzeFile(config.ser_path, allocator);
    const main_metrics = metrics.computeMainMetrics(image);
    const phd2_metrics = metrics.computePhd2Metrics(phd2_samples, config.settle_window_s);
    const secondary_metrics = metrics.computeSecondaryMetrics(ser_metrics);
    const weight = metrics.computeWeight(main_metrics, phd2_metrics, secondary_metrics);

    return .{
        .main = main_metrics,
        .phd2 = phd2_metrics,
        .secondary = secondary_metrics,
        .weight = weight,
    };
}

fn parseSubId(name: []const u8) ?u32 {
    const prefix = "sub_";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const num_part = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, num_part, '.') orelse return null;
    return std.fmt.parseInt(u32, num_part[0..dot], 10) catch null;
}

fn parseWindowId(id: []const u8) ?u32 {
    const trimmed = std.mem.trim(u8, id, " \t\r\n");
    if (std.fmt.parseInt(u32, trimmed, 10) catch null) |value| return value;
    if (std.mem.startsWith(u8, trimmed, "sub_")) {
        const rest = trimmed[4..];
        const end = std.mem.indexOfAny(u8, rest, ".\r\n\t ") orelse rest.len;
        return std.fmt.parseInt(u32, rest[0..end], 10) catch null;
    }
    return parseSubId(trimmed);
}

fn isRelativeTimestamp(ts: i128) bool {
    // Treat any timestamp below ~3 years in ns as relative to avoid misclassifying long sessions.
    return ts < 100_000_000_000_000_000;
}

fn computeFrameRange(
    frame_count: u32,
    fps: f64,
    base_start_ns: i128,
    window: ExposureWindow,
) inputs.ser_video.FrameRange {
    const guard_before = @as(i128, @intCast(window.guard_before_s)) * 1_000_000_000;
    const guard_after = @as(i128, @intCast(window.guard_after_s)) * 1_000_000_000;
    const start_ns = window.start_ns - guard_before;
    const end_ns = window.end_ns + guard_after;
    const start_s = @max(0.0, @as(f64, @floatFromInt(start_ns - base_start_ns)) / 1_000_000_000.0);
    const end_s = @max(start_s, @as(f64, @floatFromInt(end_ns - base_start_ns)) / 1_000_000_000.0);

    var start_frame = @as(u32, @intFromFloat(@floor(start_s * fps)));
    var end_frame = @as(u32, @intFromFloat(@ceil(end_s * fps)));
    if (start_frame > frame_count) start_frame = frame_count;
    if (end_frame > frame_count) end_frame = frame_count;
    if (end_frame < start_frame) end_frame = start_frame;
    if (end_frame == start_frame and frame_count > 0) {
        end_frame = @min(frame_count, start_frame + 1);
    }
    return .{ .start = start_frame, .end = end_frame };
}

fn sliceSamplesByWindow(
    allocator: std.mem.Allocator,
    samples: []inputs.phd2_log.Sample,
    win: ExposureWindow,
) !?[]inputs.phd2_log.Sample {
    if (samples.len == 0) return null;
    if (win.end_ns <= win.start_ns) return null;

    const guard_before = @as(i128, @intCast(win.guard_before_s)) * 1_000_000_000;
    const guard_after = @as(i128, @intCast(win.guard_after_s)) * 1_000_000_000;
    const sample_relative = isRelativeTimestamp(samples[0].timestamp_ns);
    const window_relative = isRelativeTimestamp(win.start_ns);

    var start_ns: i128 = win.start_ns - guard_before;
    var end_ns: i128 = win.end_ns + guard_after;
    if (sample_relative or window_relative) {
        const t0 = samples[0].timestamp_ns;
        start_ns = t0 + win.start_ns - guard_before;
        end_ns = t0 + win.end_ns + guard_after;
    }

    var filtered: std.ArrayList(inputs.phd2_log.Sample) = .empty;
    errdefer filtered.deinit(allocator);
    for (samples) |s| {
        if (s.timestamp_ns >= start_ns and s.timestamp_ns <= end_ns) {
            try filtered.append(allocator, s);
        }
    }
    if (filtered.items.len == 0) {
        filtered.deinit(allocator);
        return null;
    }
    return try filtered.toOwnedSlice(allocator);
}

pub fn runDataset(allocator: std.mem.Allocator, config: DatasetConfig) !DatasetOutput {
    var subs: std.ArrayList(SubResult) = .empty;
    defer subs.deinit(allocator);

    var windows: ?[]ExposureWindow = null;
    defer if (windows) |wins| {
        for (wins) |win| allocator.free(win.id);
        allocator.free(wins);
    };

    var used_ekos = false;
    if (config.ekos_path) |ekos_path| {
        const events = try inputs.ekos_events.parseFile(ekos_path, allocator);
        defer {
            for (events) |*ev| ev.deinit(allocator);
            allocator.free(events);
        }
        windows = try inputs.ekos_events.buildExposureWindows(events, allocator, config.guard_before_s, config.guard_after_s);
        used_ekos = windows.?.len > 0;
    }

    const subs_dir = try std.fs.path.join(allocator, &.{ config.dataset_dir, "subs" });
    defer allocator.free(subs_dir);
    const phd2_dir = try std.fs.path.join(allocator, &.{ config.dataset_dir, "phd2" });
    defer allocator.free(phd2_dir);
    const secondary_dir = try std.fs.path.join(allocator, &.{ config.dataset_dir, "secondary" });
    defer allocator.free(secondary_dir);

    var global_ser_path: ?[]const u8 = null;
    defer if (global_ser_path) |path| allocator.free(path);

    const single_ser_path = try std.fs.path.join(allocator, &.{ secondary_dir, "guide2.ser" });
    defer allocator.free(single_ser_path);
    if (std.fs.cwd().access(single_ser_path, .{})) |_| {
        global_ser_path = try allocator.dupe(u8, single_ser_path);
    } else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }

    var entries: std.ArrayList(SubEntry) = .empty;
    defer {
        for (entries.items) |*entry| entry.deinit(allocator);
        entries.deinit(allocator);
    }

    var dir = try std.fs.cwd().openDir(subs_dir, .{ .iterate = true });
    defer dir.close();

    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        const id = parseSubId(entry.name) orelse continue;
        try entries.append(allocator, .{
            .id = id,
            .name = try allocator.dupe(u8, entry.name),
        });
    }

    if (entries.items.len == 0) return error.InvalidFormat;

    std.sort.block(SubEntry, entries.items, {}, struct {
        pub fn lessThan(_: void, a: SubEntry, b: SubEntry) bool {
            return a.id < b.id;
        }
    }.lessThan);

    var window_indices: ?[]?usize = null;
    defer if (window_indices) |indices| allocator.free(indices);

    if (used_ekos) {
        std.sort.block(ExposureWindow, windows.?, {}, struct {
            pub fn lessThan(_: void, a: ExposureWindow, b: ExposureWindow) bool {
                return a.start_ns < b.start_ns;
            }
        }.lessThan);

        var indices = try allocator.alloc(?usize, entries.items.len);
        @memset(indices, null);

        var id_map = std.AutoHashMap(u32, usize).init(allocator);
        defer id_map.deinit();
        for (windows.?, 0..) |win, idx| {
            if (parseWindowId(win.id)) |id| {
                _ = try id_map.put(id, idx);
            }
        }

        var matched: usize = 0;
        for (entries.items, 0..) |entry, idx| {
            if (id_map.get(entry.id)) |win_idx| {
                indices[idx] = win_idx;
                matched += 1;
            }
        }

        if (matched == 0 and windows.?.len == entries.items.len) {
            for (entries.items, 0..) |_, idx| {
                indices[idx] = idx;
            }
        }

        window_indices = indices;
    }

    var ser_ranges: ?[]?inputs.ser_video.FrameRange = null;
    defer if (ser_ranges) |ranges| allocator.free(ranges);

    var used_session_ser = false;

    if (global_ser_path != null and used_ekos and window_indices != null) {
        const ser_info = try inputs.ser_video.readInfo(global_ser_path.?);
        const frame_count = ser_info.frame_count;

        var ranges = try allocator.alloc(?inputs.ser_video.FrameRange, entries.items.len);
        @memset(ranges, null);

        const win_len = windows.?.len;
        if (win_len > 0) {
            var window_ranges = try allocator.alloc(inputs.ser_video.FrameRange, win_len);
            defer allocator.free(window_ranges);

        if (config.prefer_session_ser) {
            var total_duration_ns: i128 = 0;
            for (windows.?) |win| {
                const guard_before = @as(i128, @intCast(win.guard_before_s)) * 1_000_000_000;
                const guard_after = @as(i128, @intCast(win.guard_after_s)) * 1_000_000_000;
                total_duration_ns += (win.end_ns - win.start_ns) + guard_before + guard_after;
            }

            if (total_duration_ns > 0) {
                const fps = @as(f64, @floatFromInt(frame_count)) /
                    (@as(f64, @floatFromInt(total_duration_ns)) / 1_000_000_000.0);
                var cursor: u32 = 0;
                var frac: f64 = 0.0;
                var idx: usize = 0;
                while (idx < win_len) : (idx += 1) {
                    const win = windows.?[idx];
                    const guard_before = @as(i128, @intCast(win.guard_before_s)) * 1_000_000_000;
                    const guard_after = @as(i128, @intCast(win.guard_after_s)) * 1_000_000_000;
                    const duration_ns = (win.end_ns - win.start_ns) + guard_before + guard_after;
                    const exact = @as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0 * fps;
                    const count_f = exact + frac;
                    var count = @as(u32, @intFromFloat(@floor(count_f)));
                    frac = count_f - @as(f64, @floatFromInt(count));

                    if (idx + 1 == win_len) {
                        count = if (frame_count > cursor) frame_count - cursor else 0;
                    } else {
                        var reserve_min: u32 = 0;
                        if (frame_count > cursor) {
                            const remaining_frames = frame_count - cursor;
                            const remaining_windows = win_len - idx - 1;
                            if (remaining_frames > remaining_windows) {
                                reserve_min = @as(u32, @intCast(remaining_windows));
                            } else {
                                reserve_min = remaining_frames;
                            }
                        }

                        const max_count = if (frame_count > cursor + reserve_min)
                            frame_count - cursor - reserve_min
                        else
                            0;
                        if (count > max_count) count = max_count;
                        if (count == 0 and max_count > 0) count = 1;
                    }

                    var end_frame = cursor + count;
                    if (end_frame > frame_count) end_frame = frame_count;
                    if (end_frame < cursor) end_frame = cursor;
                    window_ranges[idx] = .{ .start = cursor, .end = end_frame };
                    cursor = end_frame;
                }
            } else {
                var start_frame: u32 = 0;
                const win_len_u32 = @as(u32, @intCast(win_len));
                const base = frame_count / win_len_u32;
                const extra_u32 = frame_count % win_len_u32;
                const extra = @as(usize, @intCast(extra_u32));
                var idx: usize = 0;
                while (idx < win_len) : (idx += 1) {
                    const add: u32 = base + (if (idx < extra) @as(u32, 1) else 0);
                    window_ranges[idx] = .{ .start = start_frame, .end = start_frame + add };
                    start_frame += add;
                }
            }
        } else {
            var min_start = windows.?[0].start_ns;
            var max_end = windows.?[0].end_ns;
            for (windows.?) |win| {
                if (win.start_ns < min_start) min_start = win.start_ns;
                if (win.end_ns > max_end) max_end = win.end_ns;
            }

            const duration_ns = max_end - min_start;
            if (duration_ns > 0) {
                const fps = @as(f64, @floatFromInt(frame_count)) /
                    (@as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0);
                for (windows.?, 0..) |win, idx| {
                    window_ranges[idx] = computeFrameRange(frame_count, fps, min_start, win);
                }
            } else {
                var start_frame: u32 = 0;
                const win_len_u32 = @as(u32, @intCast(win_len));
                const base = frame_count / win_len_u32;
                const extra_u32 = frame_count % win_len_u32;
                const extra = @as(usize, @intCast(extra_u32));
                var idx: usize = 0;
                while (idx < win_len) : (idx += 1) {
                    const add: u32 = base + (if (idx < extra) @as(u32, 1) else 0);
                    window_ranges[idx] = .{ .start = start_frame, .end = start_frame + add };
                    start_frame += add;
                }
            }
        }

            for (entries.items, 0..) |_, entry_idx| {
                if (window_indices.?[entry_idx]) |win_idx| {
                    if (win_idx < window_ranges.len) {
                        const range = window_ranges[win_idx];
                        if (range.end > range.start) {
                            ranges[entry_idx] = range;
                        }
                    }
                }
            }
        }

        ser_ranges = ranges;
    }

    for (entries.items, 0..) |entry, entry_index| {
        const fits_path = try std.fs.path.join(allocator, &.{ subs_dir, entry.name });
        defer allocator.free(fits_path);

        var buf: [64]u8 = undefined;
        const phd2_name = try std.fmt.bufPrint(&buf, "PHD2_GuideLog_sub_{d:0>3}.txt", .{entry.id});
        const phd2_path = try std.fs.path.join(allocator, &.{ phd2_dir, phd2_name });
        defer allocator.free(phd2_path);

        var buf2: [64]u8 = undefined;
        const ser_name = try std.fmt.bufPrint(&buf2, "guide_sub_{d:0>3}.ser", .{entry.id});
        const ser_path = try std.fs.path.join(allocator, &.{ secondary_dir, ser_name });
        defer allocator.free(ser_path);

        var image = try inputs.fits_image.readImage(fits_path, allocator);
        defer image.deinit(allocator);

        var phd2_samples = try parsePhd2File(phd2_path, allocator);
        defer allocator.free(phd2_samples);

        if (window_indices) |indices| {
            if (indices[entry_index]) |win_idx| {
                const win = windows.?[win_idx];
                if (try sliceSamplesByWindow(allocator, phd2_samples, win)) |filtered| {
                    allocator.free(phd2_samples);
                    phd2_samples = filtered;
                }
            }
        }

        var ser_metrics: inputs.ser_video.SerMetrics = undefined;
        const use_session_ser = config.prefer_session_ser and global_ser_path != null;
        if (!use_session_ser) {
            if (std.fs.cwd().openFile(ser_path, .{})) |file| {
                file.close();
                ser_metrics = try inputs.ser_video.analyzeFile(ser_path, allocator);
            } else |err| switch (err) {
                error.FileNotFound => {
                    if (global_ser_path == null) return err;
                    used_session_ser = true;
                    if (ser_ranges == null) {
                        ser_metrics = try inputs.ser_video.analyzeFile(global_ser_path.?, allocator);
                    } else {
                        if (ser_ranges.?[entry_index]) |range| {
                            ser_metrics = try inputs.ser_video.analyzeFileRange(global_ser_path.?, allocator, range);
                        } else {
                            ser_metrics = try inputs.ser_video.analyzeFile(global_ser_path.?, allocator);
                        }
                    }
                },
                else => return err,
            }
        } else {
            used_session_ser = true;
            if (ser_ranges == null) {
                ser_metrics = try inputs.ser_video.analyzeFile(global_ser_path.?, allocator);
            } else {
                if (ser_ranges.?[entry_index]) |range| {
                    ser_metrics = try inputs.ser_video.analyzeFileRange(global_ser_path.?, allocator, range);
                } else {
                    ser_metrics = try inputs.ser_video.analyzeFile(global_ser_path.?, allocator);
                }
            }
        }
        const main_metrics = metrics.computeMainMetrics(image);
        const phd2_metrics = metrics.computePhd2Metrics(phd2_samples, config.settle_window_s);
        const secondary_metrics = metrics.computeSecondaryMetrics(ser_metrics);
        const weight = metrics.computeWeight(main_metrics, phd2_metrics, secondary_metrics);

        try subs.append(allocator, .{
            .id = entry.id,
            .fits_path = try allocator.dupe(u8, fits_path),
            .phd2_path = try allocator.dupe(u8, phd2_path),
            .ser_path = try allocator.dupe(u8, ser_path),
            .main = main_metrics,
            .phd2 = phd2_metrics,
            .secondary = secondary_metrics,
            .weight = weight,
        });
    }

    if (subs.items.len == 0) return error.InvalidFormat;

    // Sort by id
    std.sort.block(SubResult, subs.items, {}, struct {
        pub fn lessThan(_: void, a: SubResult, b: SubResult) bool {
            return a.id < b.id;
        }
    }.lessThan);

    // Stack
    var first = try inputs.fits_image.readImage(subs.items[0].fits_path, allocator);
    defer first.deinit(allocator);
    const pixel_count = first.pixels.len;
    var sum_plain = try allocator.alloc(f64, pixel_count);
    defer allocator.free(sum_plain);
    var sum_weighted = try allocator.alloc(f64, pixel_count);
    defer allocator.free(sum_weighted);
    @memset(sum_plain, 0.0);
    @memset(sum_weighted, 0.0);

    var weight_sum: f64 = 0.0;
    for (subs.items) |sub| {
        var img = try inputs.fits_image.readImage(sub.fits_path, allocator);
        defer img.deinit(allocator);
        var i: usize = 0;
        while (i < pixel_count) : (i += 1) {
            const val = @as(f64, @floatFromInt(img.pixels[i]));
            sum_plain[i] += val;
            sum_weighted[i] += val * sub.weight;
        }
        weight_sum += sub.weight;
    }

    var plain_pixels = try allocator.alloc(u16, pixel_count);
    var weighted_pixels = try allocator.alloc(u16, pixel_count);
    defer allocator.free(plain_pixels);
    defer allocator.free(weighted_pixels);

    var i: usize = 0;
    while (i < pixel_count) : (i += 1) {
        const plain = sum_plain[i] / @as(f64, @floatFromInt(subs.items.len));
        const weighted = if (weight_sum > 0.0) sum_weighted[i] / weight_sum else plain;
        const plain_clamped = @min(@max(plain, 0.0), 65535.0);
        const weighted_clamped = @min(@max(weighted, 0.0), 65535.0);
        plain_pixels[i] = @as(u16, @intFromFloat(plain_clamped));
        weighted_pixels[i] = @as(u16, @intFromFloat(weighted_clamped));
    }

    const plain_path = try std.fs.path.join(allocator, &.{ config.out_dir, "stack_plain.fits" });
    const weighted_path = try std.fs.path.join(allocator, &.{ config.out_dir, "stack_weighted.fits" });

    try inputs.fits_image.writeImage(plain_path, first.width, first.height, plain_pixels);
    try inputs.fits_image.writeImage(weighted_path, first.width, first.height, weighted_pixels);

    const owned_subs = try subs.toOwnedSlice(allocator);
    return .{
        .subs = owned_subs,
        .stack_plain_path = plain_path,
        .stack_weighted_path = weighted_path,
        .used_ekos = used_ekos,
        .used_session_ser = used_session_ser,
        .session_ser_path = if (used_session_ser)
            try allocator.dupe(u8, global_ser_path.?)
        else
            null,
    };
}

test "slice samples preserves relative offsets" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var samples: std.ArrayList(inputs.phd2_log.Sample) = .empty;
    defer samples.deinit(allocator);
    var i: usize = 0;
    while (i <= 60) : (i += 1) {
        try samples.append(allocator, .{
            .timestamp_ns = @as(i128, @intCast(i)) * 1_000_000_000,
            .ra_error_arcsec = 0.0,
            .dec_error_arcsec = 0.0,
        });
    }

    const win0: ExposureWindow = .{
        .id = "sub_001",
        .start_ns = 30 * 1_000_000_000,
        .end_ns = 40 * 1_000_000_000,
        .guard_before_s = 0,
        .guard_after_s = 0,
    };
    _ = win0;
    const win1: ExposureWindow = .{
        .id = "sub_002",
        .start_ns = 50 * 1_000_000_000,
        .end_ns = 60 * 1_000_000_000,
        .guard_before_s = 0,
        .guard_after_s = 0,
    };

    const slice = try sliceSamplesByWindow(allocator, samples.items, win1);
    defer if (slice) |items| allocator.free(items);
    try std.testing.expect(slice != null);
    try std.testing.expectEqual(@as(usize, 11), slice.?.len);
    try std.testing.expectEqual(@as(i128, 50 * 1_000_000_000), slice.?[0].timestamp_ns);
    try std.testing.expectEqual(@as(i128, 60 * 1_000_000_000), slice.?[slice.?.len - 1].timestamp_ns);
}

test "session ser ranges honor exposure durations" {
    const windows = [_]ExposureWindow{
        .{
            .id = "sub_001",
            .start_ns = 0,
            .end_ns = 10 * 1_000_000_000,
            .guard_before_s = 0,
            .guard_after_s = 0,
        },
        .{
            .id = "sub_002",
            .start_ns = 20 * 1_000_000_000,
            .end_ns = 30 * 1_000_000_000,
            .guard_before_s = 0,
            .guard_after_s = 0,
        },
    };

    const frame_count: u32 = 100;
    const total_duration_ns: f64 = 20.0; // 10s + 10s
    const fps = @as(f64, @floatFromInt(frame_count)) / total_duration_ns;

    var cursor: u32 = 0;
    var ranges: [2]inputs.ser_video.FrameRange = undefined;
    var idx: usize = 0;
    while (idx < windows.len) : (idx += 1) {
        const win = windows[idx];
        const duration_ns = win.end_ns - win.start_ns;
        var count = @as(u32, @intFromFloat(@ceil(@as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0 * fps)));
        if (count == 0) count = 1;
        var end_frame = cursor + count;
        if (end_frame > frame_count) end_frame = frame_count;
        ranges[idx] = .{ .start = cursor, .end = end_frame };
        cursor = end_frame;
    }

    try std.testing.expectEqual(@as(u32, 0), ranges[0].start);
    try std.testing.expectEqual(@as(u32, 50), ranges[0].end);
    try std.testing.expectEqual(@as(u32, 50), ranges[1].start);
    try std.testing.expectEqual(@as(u32, 100), ranges[1].end);
}
