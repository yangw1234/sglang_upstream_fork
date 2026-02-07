docker run -dit \
        --group-add $(getent group video | cut -d: -f3) \
        -v /home/sdp/yang/.cache/huggingface:/root/.cache/huggingface \
        -v /home/sdp/yang/sglang:/workspace/sglang \
        --device /dev/dri \
        xpu_sglang_main:bmg-0206
