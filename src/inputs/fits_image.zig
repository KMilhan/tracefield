const std = @import("std");

pub const FitsImage = struct {
    width: usize,
    height: usize,
    pixels: []u16,

    pub fn deinit(self: *FitsImage, allocator: std.mem.Allocator) void {
        allocator.free(self.pixels);
        self.* = undefined;
    }
};

pub const ParseError = std.mem.Allocator.Error ||
    std.fs.File.OpenError ||
    std.fs.File.ReadError ||
    std.fs.File.SeekError ||
    std.fs.File.GetEndPosError ||
    error{EndOfStream} ||
    error{
        InvalidHeader,
        MissingKeyword,
        UnsupportedBitpix,
        UnsupportedAxis,
    };

const block_size = 2880;
const card_size = 80;

const FitsHeader = struct {
    bitpix: i32,
    naxis: i32,
    naxis1: usize,
    naxis2: usize,
    data_offset: usize,
    bzero: f64,
    bscale: f64,
};

fn parseIntValue(card: []const u8) ?i64 {
    if (card.len < 10 or card[8] != '=') return null;
    var value_slice = std.mem.trim(u8, card[10..], " \t");
    if (std.mem.indexOfScalar(u8, value_slice, '/')) |idx| {
        value_slice = std.mem.trim(u8, value_slice[0..idx], " \t");
    }
    return std.fmt.parseInt(i64, value_slice, 10) catch null;
}

fn parseFloatValue(card: []const u8) ?f64 {
    if (card.len < 10 or card[8] != '=') return null;
    var value_slice = std.mem.trim(u8, card[10..], " \t");
    if (std.mem.indexOfScalar(u8, value_slice, '/')) |idx| {
        value_slice = std.mem.trim(u8, value_slice[0..idx], " \t");
    }
    return std.fmt.parseFloat(f64, value_slice) catch null;
}

fn readHeader(file: *std.fs.File) ParseError!FitsHeader {
    var reader = file.deprecatedReader();
    var bitpix: ?i32 = null;
    var naxis: ?i32 = null;
    var naxis1: ?usize = null;
    var naxis2: ?usize = null;
    var offset: usize = 0;
    var bzero: f64 = 0.0;
    var bscale: f64 = 1.0;
    var end_found = false;

    var block: [block_size]u8 = undefined;
    while (!end_found) {
        try reader.readNoEof(&block);
        var i: usize = 0;
        while (i < block_size / card_size) : (i += 1) {
            const card = block[i * card_size .. (i + 1) * card_size];
            const key = std.mem.trimRight(u8, card[0..8], " ");
            if (std.mem.eql(u8, key, "END")) {
                end_found = true;
                offset += (i + 1) * card_size;
                break;
            }
            if (std.mem.eql(u8, key, "BITPIX")) {
                if (parseIntValue(card)) |val| bitpix = @intCast(val);
            } else if (std.mem.eql(u8, key, "NAXIS")) {
                if (parseIntValue(card)) |val| naxis = @intCast(val);
            } else if (std.mem.eql(u8, key, "NAXIS1")) {
                if (parseIntValue(card)) |val| naxis1 = @intCast(val);
            } else if (std.mem.eql(u8, key, "NAXIS2")) {
                if (parseIntValue(card)) |val| naxis2 = @intCast(val);
            } else if (std.mem.eql(u8, key, "BZERO")) {
                if (parseFloatValue(card)) |val| bzero = val;
            } else if (std.mem.eql(u8, key, "BSCALE")) {
                if (parseFloatValue(card)) |val| bscale = val;
            }
        }
        if (!end_found) offset += block_size;
    }

    if (bitpix == null or naxis == null or naxis1 == null or naxis2 == null) {
        return error.MissingKeyword;
    }
    if (bitpix.? != 16) return error.UnsupportedBitpix;
    if (naxis.? != 2) return error.UnsupportedAxis;

    const data_offset = ((offset + block_size - 1) / block_size) * block_size;

    return .{
        .bitpix = bitpix.?,
        .naxis = naxis.?,
        .naxis1 = naxis1.?,
        .naxis2 = naxis2.?,
        .data_offset = data_offset,
        .bzero = bzero,
        .bscale = bscale,
    };
}

pub fn readImage(path: []const u8, allocator: std.mem.Allocator) ParseError!FitsImage {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const header = try readHeader(&file);
    try file.seekTo(header.data_offset);

    var reader = file.deprecatedReader();

    const pixel_count = header.naxis1 * header.naxis2;
    var pixels = try allocator.alloc(u16, pixel_count);
    errdefer allocator.free(pixels);

    const buf = try allocator.alloc(u8, pixel_count * 2);
    defer allocator.free(buf);
    try reader.readNoEof(buf);

    var idx: usize = 0;
    var offset: usize = 0;
    while (idx < pixel_count) : (idx += 1) {
        var two: [2]u8 = .{ buf[offset], buf[offset + 1] };
        const raw = std.mem.readInt(i16, &two, .big);
        const value = (@as(f64, @floatFromInt(raw)) * header.bscale) + header.bzero;
        const clamped = @min(@max(value, 0.0), 65535.0);
        pixels[idx] = @as(u16, @intFromFloat(clamped));
        offset += 2;
    }

    return .{
        .width = header.naxis1,
        .height = header.naxis2,
        .pixels = pixels,
    };
}

fn writeCard(writer: anytype, key: []const u8, value: []const u8) !void {
    var card: [card_size]u8 = [_]u8{' '} ** card_size;
    const key_len = @min(key.len, 8);
    @memcpy(card[0..key_len], key[0..key_len]);
    if (value.len > 0) {
        card[8] = '=';
        card[9] = ' ';
        const val_len = @min(value.len, card_size - 10);
        @memcpy(card[10 .. 10 + val_len], value[0..val_len]);
    }
    try writer.writeAll(&card);
}

fn writeIntCard(writer: anytype, key: []const u8, value: i64) !void {
    var buf: [32]u8 = undefined;
    const raw = try std.fmt.bufPrint(&buf, "{d}", .{value});
    var value_buf: [20]u8 = [_]u8{' '} ** 20;
    const start = value_buf.len - raw.len;
    @memcpy(value_buf[start..], raw);
    try writeCard(writer, key, value_buf[0..]);
}

fn writeFloatCard(writer: anytype, key: []const u8, value: f64) !void {
    var buf: [32]u8 = undefined;
    const raw = try std.fmt.bufPrint(&buf, "{d}", .{value});
    var value_buf: [20]u8 = [_]u8{' '} ** 20;
    const start = value_buf.len - raw.len;
    @memcpy(value_buf[start..], raw);
    try writeCard(writer, key, value_buf[0..]);
}

fn writeBoolCard(writer: anytype, key: []const u8, value: bool) !void {
    const raw = if (value) "T" else "F";
    var value_buf: [20]u8 = [_]u8{' '} ** 20;
    const start = value_buf.len - 1;
    value_buf[start] = raw[0];
    try writeCard(writer, key, value_buf[0..]);
}

pub fn writeImage(path: []const u8, width: usize, height: usize, pixels: []const u16) !void {
    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    var writer = file.deprecatedWriter();

    try writeBoolCard(writer, "SIMPLE", true);
    try writeIntCard(writer, "BITPIX", 16);
    try writeIntCard(writer, "NAXIS", 2);
    try writeIntCard(writer, "NAXIS1", @intCast(width));
    try writeIntCard(writer, "NAXIS2", @intCast(height));
    try writeIntCard(writer, "BZERO", 32768);
    try writeFloatCard(writer, "BSCALE", 1.0);
    try writeCard(writer, "END", "");

    // pad header to 2880
    const header_bytes = 8 * card_size;
    const pad = (block_size - (header_bytes % block_size)) % block_size;
    if (pad > 0) {
        var pad_buf: [block_size]u8 = [_]u8{' '} ** block_size;
        try writer.writeAll(pad_buf[0..pad]);
    }

    // write data as signed i16 with BZERO=32768 in big-endian
    var i: usize = 0;
    var buf: [2]u8 = undefined;
    while (i < pixels.len) : (i += 1) {
        const v = pixels[i];
        const signed = @as(i32, @intCast(v)) - 32768;
        const raw = @as(i16, @intCast(@max(@min(signed, 32767), -32768)));
        const u = @as(u16, @bitCast(raw));
        buf[0] = @as(u8, @intCast((u >> 8) & 0xFF));
        buf[1] = @as(u8, @intCast(u & 0xFF));
        try writer.writeAll(&buf);
    }

    // pad data to 2880
    const data_bytes = pixels.len * 2;
    const data_pad = (block_size - (data_bytes % block_size)) % block_size;
    if (data_pad > 0) {
        var pad_buf: [block_size]u8 = [_]u8{0} ** block_size;
        try writer.writeAll(pad_buf[0..data_pad]);
    }
}

test "read fits header" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    var image = try readImage("testdata/main/main_4k.fits", allocator);
    defer image.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 4096), image.width);
    try std.testing.expectEqual(@as(usize, 4096), image.height);
}
