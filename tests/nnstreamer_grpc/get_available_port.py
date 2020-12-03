#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2020 Samsung Electronics
#
# @file get_available_port.py
# @brief Get available socket port in local machine
# @author Dongju Chae <dongju.chae@samsung.com>

import socket

if __name__ == "__main__":
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  # Bind to any available port
  s.bind(('', 0))
  addr = s.getsockname()
  s.close()

  print (addr[1])
