# coding: utf-8
# Copyright @tongshiwei

from longling import config_logging

config_logging(logger="RFAG", console_log_level="info")

from .RFAG import RFAG

__all__ = ["RFAG"]
