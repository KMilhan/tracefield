const std = @import("std");
const core = @import("tracefield_core");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var config = core.pipeline.RunConfig{
        .fits_path = "testdata/main/main_4k.fits",
        .phd2_path = "testdata/phd2/PHD2_GuideLog_2026-02-01_210500.txt",
        .ser_path = "testdata/secondary/guide2.ser",
        .settle_window_s = 5.0,
    };

    var dataset_dir: ?[]const u8 = null;
    var out_dir: ?[]const u8 = null;
    var settle_window_s: f64 = 5.0;
    var ekos_path: ?[]const u8 = null;
    var guard_before_s: u32 = 0;
    var guard_after_s: u32 = 0;
    var prefer_session_ser = false;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help")) {
            const stdout = std.fs.File.stdout().deprecatedWriter();
            try stdout.print(
                "usage: tracefield [--fits PATH] [--phd2 PATH] [--ser PATH] [--dataset DIR] [--out DIR] [--ekos PATH] [--guard-before S] [--guard-after S] [--settle S] [--prefer-session-ser]\n",
                .{},
            );
            return;
        } else if (std.mem.eql(u8, arg, "--fits") and i + 1 < args.len) {
            i += 1;
            config.fits_path = args[i];
        } else if (std.mem.eql(u8, arg, "--phd2") and i + 1 < args.len) {
            i += 1;
            config.phd2_path = args[i];
        } else if (std.mem.eql(u8, arg, "--ser") and i + 1 < args.len) {
            i += 1;
            config.ser_path = args[i];
        } else if (std.mem.eql(u8, arg, "--dataset") and i + 1 < args.len) {
            i += 1;
            dataset_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--out") and i + 1 < args.len) {
            i += 1;
            out_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--settle") and i + 1 < args.len) {
            i += 1;
            settle_window_s = std.fmt.parseFloat(f64, args[i]) catch settle_window_s;
        } else if (std.mem.eql(u8, arg, "--ekos") and i + 1 < args.len) {
            i += 1;
            ekos_path = args[i];
        } else if (std.mem.eql(u8, arg, "--guard-before") and i + 1 < args.len) {
            i += 1;
            guard_before_s = std.fmt.parseInt(u32, args[i], 10) catch guard_before_s;
        } else if (std.mem.eql(u8, arg, "--guard-after") and i + 1 < args.len) {
            i += 1;
            guard_after_s = std.fmt.parseInt(u32, args[i], 10) catch guard_after_s;
        } else if (std.mem.eql(u8, arg, "--prefer-session-ser")) {
            prefer_session_ser = true;
        }
    }

    if (dataset_dir) |dataset| {
        const out_path = out_dir orelse dataset;
        const data_config = core.pipeline.DatasetConfig{
            .dataset_dir = dataset,
            .out_dir = out_path,
            .settle_window_s = settle_window_s,
            .ekos_path = ekos_path,
            .guard_before_s = guard_before_s,
            .guard_after_s = guard_after_s,
            .prefer_session_ser = prefer_session_ser,
        };
        var output = try core.pipeline.runDataset(allocator, data_config);
        const json = try std.json.Stringify.valueAlloc(allocator, output, .{ .whitespace = .indent_2 });
        defer allocator.free(json);
        try std.fs.File.stdout().writeAll(json);
        try std.fs.File.stdout().writeAll("\n");
        output.deinit(allocator);
        return;
    }

    config.settle_window_s = settle_window_s;
    const output = try core.pipeline.run(allocator, config);
    const json = try std.json.Stringify.valueAlloc(allocator, output, .{ .whitespace = .indent_2 });
    defer allocator.free(json);
    try std.fs.File.stdout().writeAll(json);
    try std.fs.File.stdout().writeAll("\n");
}

test "basic" {
    try std.testing.expect(true);
}
