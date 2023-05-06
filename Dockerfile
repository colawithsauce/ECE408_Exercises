FROM nvidia/cuda:12.1.1-devel-ubi8
RUN dnf install -y cmake
ENV http_proxy=http://127.0.0.1:7890
ENV https_proxy=http://127.0.0.1:7890
ENV all_proxy=http://127.0.0.1:7890

