docker build --no-cache \
--build-arg http_proxy=http://proxy-dmz.intel.com:912 \
--build-arg https_proxy=http://proxy-dmz.intel.com:912 \
--build-arg no_proxy=.intel.com,intel.com,localhost,127.0.0.1 \
--progress=plain -f docker/xpu.Dockerfile -t xpu_sglang_main:bmg-0206 . 