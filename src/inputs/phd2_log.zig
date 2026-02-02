const std = @import("std");

pub const Sample = struct {
    timestamp_ns: i128,
    ra_error_arcsec: f64,
    dec_error_arcsec: f64,
};

pub const ParseError = std.mem.Allocator.Error || error{
    InvalidFormat,
    InvalidTimestamp,
    InvalidNumber,
    MissingColumns,
};

pub fn parseLine(line: []const u8) ParseError!Sample {
    // Expected format: "<timestamp_ns>,<ra_error_arcsec>,<dec_error_arcsec>"
    var it = std.mem.splitScalar(u8, line, ',');
    const ts_str = it.next() orelse return error.InvalidFormat;
    const ra_str = it.next() orelse return error.InvalidFormat;
    const dec_str = it.next() orelse return error.InvalidFormat;
    if (it.next() != null) return error.InvalidFormat;

    const ts = std.fmt.parseInt(i128, std.mem.trim(u8, ts_str, " \t\r\n"), 10) catch {
        return error.InvalidTimestamp;
    };

    const ra = std.fmt.parseFloat(f64, std.mem.trim(u8, ra_str, " \t\r\n")) catch {
        return error.InvalidNumber;
    };

    const dec = std.fmt.parseFloat(f64, std.mem.trim(u8, dec_str, " \t\r\n")) catch {
        return error.InvalidNumber;
    };

    return .{
        .timestamp_ns = ts,
        .ra_error_arcsec = ra,
        .dec_error_arcsec = dec,
    };
}

pub const GuideLog = struct {
    samples: []Sample,
    pub fn deinit(self: *GuideLog, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
        self.* = undefined;
    }
};

pub fn parseGuideLog(content: []const u8, allocator: std.mem.Allocator) ParseError!GuideLog {
    // Expected: header lines then CSV header including Time,RARawDistance,DECRawDistance
    var line_it = std.mem.splitScalar(u8, content, '\n');
    var header_found = false;
    var time_idx: ?usize = null;
    var ra_idx: ?usize = null;
    var dec_idx: ?usize = null;

    var samples: std.ArrayList(Sample) = .empty;
    errdefer samples.deinit(allocator);

    while (line_it.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;

        if (!header_found) {
            if (std.mem.startsWith(u8, line, "Frame,")) {
                header_found = true;
                var cols_it = std.mem.splitScalar(u8, line, ',');
                var idx: usize = 0;
                while (cols_it.next()) |col_raw| : (idx += 1) {
                    const col = std.mem.trim(u8, col_raw, " \t\r\"");
                    if (std.mem.eql(u8, col, "Time")) time_idx = idx;
                    if (std.mem.eql(u8, col, "RARawDistance")) ra_idx = idx;
                    if (std.mem.eql(u8, col, "DECRawDistance")) dec_idx = idx;
                    if (std.mem.eql(u8, col, "dx") and ra_idx == null) ra_idx = idx;
                    if (std.mem.eql(u8, col, "dy") and dec_idx == null) dec_idx = idx;
                }
            }
            continue;
        }

        var cols = std.mem.splitScalar(u8, line, ',');
        var idx: usize = 0;
        var time_str: ?[]const u8 = null;
        var ra_str: ?[]const u8 = null;
        var dec_str: ?[]const u8 = null;

        while (cols.next()) |col_raw| : (idx += 1) {
            const col = std.mem.trim(u8, col_raw, " \t\r\"");
            if (time_idx != null and idx == time_idx.?) time_str = col;
            if (ra_idx != null and idx == ra_idx.?) ra_str = col;
            if (dec_idx != null and idx == dec_idx.?) dec_str = col;
        }

        if (time_str == null or ra_str == null or dec_str == null) {
            return error.MissingColumns;
        }

        const time_s = std.fmt.parseFloat(f64, time_str.?) catch return error.InvalidNumber;
        const ts = @as(i128, @intFromFloat(time_s * 1_000_000_000.0));
        const ra = std.fmt.parseFloat(f64, ra_str.?) catch return error.InvalidNumber;
        const dec = std.fmt.parseFloat(f64, dec_str.?) catch return error.InvalidNumber;

        try samples.append(allocator, .{
            .timestamp_ns = ts,
            .ra_error_arcsec = ra,
            .dec_error_arcsec = dec,
        });
    }

    if (samples.items.len == 0) return error.InvalidFormat;

    return .{ .samples = try samples.toOwnedSlice(allocator) };
}

test "parse phd2 sample" {
    const sample = try parseLine("123,0.45,-0.12");
    try std.testing.expectEqual(@as(i128, 123), sample.timestamp_ns);
}

test "parse phd2 guide log" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var file = try std.fs.cwd().openFile("testdata/phd2/PHD2_GuideLog_2026-02-01_210500.txt", .{});
    defer file.close();
    const data = try file.readToEndAlloc(allocator, 2 * 1024 * 1024);
    defer allocator.free(data);

    var log = try parseGuideLog(data, allocator);
    defer log.deinit(allocator);
    try std.testing.expect(log.samples.len > 0);
}
