#!/bin/bash
tensorboard --logdir userdata/models --port 6006 --bind_all &
exec uvicorn app:app --host 0.0.0.0 --port 7860
