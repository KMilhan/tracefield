const std = @import("std");

pub const FrameMeta = struct {
    frame_index: u64,
    timestamp_ns: i128,
};

pub const ParseError = error{
    InvalidFormat,
    InvalidTimestamp,
    InvalidFrameIndex,
};

pub fn parseSidecarLine(line: []const u8) ParseError!FrameMeta {
    // Expected format: "<frame_index>,<timestamp_ns>"
    var it = std.mem.splitScalar(u8, line, ',');
    const idx_str = it.next() orelse return error.InvalidFormat;
    const ts_str = it.next() orelse return error.InvalidFormat;
    if (it.next() != null) return error.InvalidFormat;

    const idx = std.fmt.parseInt(u64, std.mem.trim(u8, idx_str, " \t\r\n"), 10) catch {
        return error.InvalidFrameIndex;
    };

    const ts = std.fmt.parseInt(i128, std.mem.trim(u8, ts_str, " \t\r\n"), 10) catch {
        return error.InvalidTimestamp;
    };

    return .{
        .frame_index = idx,
        .timestamp_ns = ts,
    };
}

// test "parse telemetry frame" {
//     const frame = try parseSidecarLine("42,123456");
//     try std.testing.expectEqual(@as(u64, 42), frame.frame_index);
// }
