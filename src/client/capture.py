import logging

import cv2

from src.common.frame import ERFrame, FrameError

_log = logging.getLogger(__name__)


class CameraError(Exception):
    def __init__(self, device: int):
        super().__init__(f"Cannot open camera device {device}")


class ERVideoCapture:
    def __init__(self, cam_idx: int | str = 0):
        self.cam_idx = cam_idx
        self.capture = cv2.VideoCapture(cam_idx)
        if not self.capture.isOpened():
            raise CameraError(cam_idx)

        self.length = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

        if self.length > 0:
            _log.info("Video length: %d", self.length)
        _log.info("Video info: size: %dx%d, FPS: %d", self.width, self.height, self.fps)

    def get_frame(self) -> ERFrame:
        """Capture frame-by-frame."""
        _ret, frame = self.capture.read()
        if not _ret:
            raise FrameError(self.cam_idx)
        _log.debug("Frame read: %s", frame.shape)
        return ERFrame(frame)

    def release(self):
        """Release the camera when done."""
        self.capture.release()
        cv2.destroyAllWindows()
