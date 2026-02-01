const std = @import("std");

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
