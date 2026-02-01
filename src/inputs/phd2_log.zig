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

// test "parse phd2 sample" {
//     const sample = try parseLine("123,0.45,-0.12");
//     try std.testing.expectEqual(@as(i128, 123), sample.timestamp_ns);
// }
