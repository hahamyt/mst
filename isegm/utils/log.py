import io
import time
import logging
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw

LOGGER_NAME = 'root'
LOGGER_DATEFMT = '%Y-%m-%d %H:%M:%S'

handler = logging.StreamHandler()

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def add_logging(logs_path, prefix):
    log_name = prefix + datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S') + '.log'
    stdout_log_path = logs_path / log_name

    fh = logging.FileHandler(str(stdout_log_path))
    formatter = logging.Formatter(fmt='(%(levelname)s) %(asctime)s: %(message)s',
                                  datefmt=LOGGER_DATEFMT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None, mininterval=20):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self.mininterval = mininterval
        self.last_time = 0

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        if len(self.buf) > 0 and time.time() - self.last_time > self.mininterval:
            self.logger.log(self.level, self.buf)
            self.last_time = time.time()


class SummaryWriterAvg(SummaryWriter):
    def __init__(self, *args, dump_period=20, **kwargs):
        super().__init__(*args, **kwargs)
        self._dump_period = dump_period
        self._avg_scalars = dict()

    def add_scalar(self, tag, value, global_step=None, disable_avg=False):
        if disable_avg or isinstance(value, (tuple, list, dict)):
            super().add_scalar(tag, np.array(value), global_step=global_step)
        else:
            if tag not in self._avg_scalars:
                self._avg_scalars[tag] = ScalarAccumulator(self._dump_period)
            avg_scalar = self._avg_scalars[tag]
            avg_scalar.add(value)

            if avg_scalar.is_full():
                super().add_scalar(tag, avg_scalar.value,
                                   global_step=global_step)
                avg_scalar.reset()


class ScalarAccumulator(object):
    def __init__(self, period):
        self.sum = 0
        self.cnt = 0
        self.period = period

    def add(self, value):
        self.sum += value
        self.cnt += 1

    @property
    def value(self):
        if self.cnt > 0:
            return self.sum / self.cnt
        else:
            return 0

    def reset(self):
        self.cnt = 0
        self.sum = 0

    def is_full(self):
        return self.cnt >= self.period

    def __len__(self):
        return self.cnt

def tensor2pillow(input_tensor: torch.Tensor):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)

    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = (input_tensor * 255)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze(0)
    if input_tensor.shape[0] == 1:
        input_tensor = input_tensor.repeat(3, 1, 1)
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)

    return im

def plt_click_patch(image, index_list, patch_sz, patch_num, color='red'):
    im = image.copy()
    a = ImageDraw.ImageDraw(im)

    for index in index_list:
        if index >= 0:
            x, y = np.unravel_index(index.cpu(), (patch_num, patch_num))
            x = x * patch_sz
            y = y * patch_sz
            w = x + patch_sz
            h = y + patch_sz

            a.rectangle([(y, x), (h, w)], fill=color, outline=color, width=4)

    return im

def plt_grid(image, patch_sz):
    im = image.copy()
    a = ImageDraw.ImageDraw(im)

    for x1 in range(0, im.height-1, patch_sz):
        a.line(((x1, 0), (x1, im.height-1)), (125, 125, 125))
    for y1 in range(0, im.height-1, patch_sz):
        a.line(((0, y1), (im.height-1, y1)), (125, 125, 125))

    return im