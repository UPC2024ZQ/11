import struct
import numpy as np
import cv2

class LightweightProcessModule:
    def __init__(self):
        self.proto_head_origin = 20
        self.proto_head_compress = 8
        self.max_distortion = 0.005
    def layer_compress(self, frame_data, stream_data):
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        compress_frame = cv2.imencode('.h265', frame)[1].tobytes()
        if hasattr(self, 'prev_frame'):
            diff = cv2.absdiff(frame, self.prev_frame)
            diff_ratio = np.count_nonzero(diff) / (frame.shape[0] * frame.shape[1])
            if diff_ratio < 0.1:
                stream_data = diff.tobytes()
        self.prev_frame = frame
        stream_data = b"".join([b for i, b in enumerate(stream_data) if i % 4 != 0])
        distortion = 1 - len(compress_frame) / len(frame_data)
        if distortion > self.max_distortion:
            return frame_data
        return compress_frame + stream_data
    def proto_head_simplify(self, packet):
        version = struct.unpack('H', packet[0:2])[0]
        data_len = struct.unpack('I', packet[2:6])[0]
        check = struct.unpack('H', packet[6:8])[0]
        compress_head = struct.pack('HIH', version, data_len, check)
        compress_packet = compress_head + packet[self.proto_head_origin:]
        return compress_packet
    def optimize_stream_fluctuation(self, stream_list):
        avg_size = np.mean([len(s) for s in stream_list])
        optimize_stream = []
        for s in stream_list:
            if len(s) > 1.5 * avg_size:
                split = [s[i:i+int(avg_size)] for i in range(0, len(s), int(avg_size))]
                optimize_stream.extend(split)
            elif len(s) < 0.5 * avg_size:
                if optimize_stream and len(optimize_stream[-1]) < 1.2 * avg_size:
                    optimize_stream[-1] += s
                else:
                    optimize_stream.append(s)
            else:
                optimize_stream.append(s)
        return optimize_stream
    def run(self, frame_data, stream_data, packet_list):
        compress_stream = self.layer_compress(frame_data, stream_data)
        packet = struct.pack('I', len(compress_stream)) + compress_stream
        compress_packet = self.proto_head_simplify(packet)
        final_packet = self.optimize_stream_fluctuation([compress_packet] + packet_list)
        return final_packet
if __name__ == "__main__":
    lightweight_module = LightweightProcessModule()
    test_frame = cv2.imencode('.h265', np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    test_stream = b'encode_stream_data' * 100
    test_packets = [b'packet_' + str(i).encode() * 50 for i in range(5)]
    result = lightweight_module.run(test_frame, test_stream, test_packets)
    print(f"轻量化处理完成，输出数据包数量：{len(result)}")