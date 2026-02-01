const std = @import("std");

pub const EventType = enum {
    exposure_start,
    exposure_end,
};

pub const Event = struct {
    timestamp_ns: i128,
    sequence_id: []const u8,
    event_type: EventType,

    pub fn deinit(self: *Event, allocator: std.mem.Allocator) void {
        allocator.free(self.sequence_id);
        self.* = undefined;
    }
};

pub const ParseError = std.mem.Allocator.Error || error{
    InvalidFormat,
    InvalidTimestamp,
    UnknownEvent,
};

pub fn parseLine(line: []const u8, allocator: std.mem.Allocator) ParseError!Event {
    // Expected format: "<timestamp_ns>,<sequence_id>,<start|end>"
    var it = std.mem.splitScalar(u8, line, ',');
    const ts_str = it.next() orelse return error.InvalidFormat;
    const seq_str = it.next() orelse return error.InvalidFormat;
    const event_str = it.next() orelse return error.InvalidFormat;
    if (it.next() != null) return error.InvalidFormat;

    const ts = std.fmt.parseInt(i128, std.mem.trim(u8, ts_str, " \t\r\n"), 10) catch {
        return error.InvalidTimestamp;
    };

    const seq_trim = std.mem.trim(u8, seq_str, " \t\r\n");
    const seq = allocator.dupe(u8, seq_trim) catch return error.OutOfMemory;
    errdefer allocator.free(seq);

    const event_trim = std.mem.trim(u8, event_str, " \t\r\n");
    const event_type = if (std.mem.eql(u8, event_trim, "start"))
        EventType.exposure_start
    else if (std.mem.eql(u8, event_trim, "end"))
        EventType.exposure_end
    else
        return error.UnknownEvent;

    return .{
        .timestamp_ns = ts,
        .sequence_id = seq,
        .event_type = event_type,
    };
}

pub fn parseLineOwned(line: []const u8, allocator: std.mem.Allocator) ParseError!Event {
    return parseLine(line, allocator);
}

// Minimal sanity check for scaffolding.
// test "parse ekos event" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     defer _ = gpa.deinit();
//     const allocator = gpa.allocator();
//
//     var ev = try parseLine("123,sub-001,start", allocator);
//     defer ev.deinit(allocator);
//     try std.testing.expectEqual(@as(i128, 123), ev.timestamp_ns);
// }
