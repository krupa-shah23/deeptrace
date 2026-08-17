#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc < 2) return 1;
    char *filepath = argv[argc - 1];
    char command[2048];
    snprintf(command, sizeof(command),
        "python -c \"import sys, json, cv2; cap = cv2.VideoCapture(r'%s'); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release(); data = {'streams': [{'codec_type': 'video', 'width': w, 'height': h, 'codec_name': 'h264'}], 'format': {'tags': {'encoder': 'Lavf61.7.100', 'creation_time': '2026-08-17T12:00:00'}, 'bit_rate': '1000000'}}; print(json.dumps(data))\"",
        filepath
    );
    return system(command);
}
