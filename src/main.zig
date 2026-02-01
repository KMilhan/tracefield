const std = @import("std");
const core = @import("tracefield_core");

pub fn main() !void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    try stdout.print("tracefield scaffold ready\n", .{});
    try core.pipeline.describe(stdout);
}

test "basic" {
    try std.testing.expect(true);
}
