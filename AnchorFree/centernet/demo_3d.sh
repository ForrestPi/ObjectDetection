if [ $# -eq 0 ];then
    echo "No input video specificed. run as ./demo_det.sh path/to/video.mp4"
else
    python3 demo.py ddd \
    --demo $1 \
    --load_model ./weights/ddd_mytrained.pth
fi

