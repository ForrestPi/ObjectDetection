if [ $# -eq 0 ];then
    echo "No input video specificed. run as ./demo_det.sh path/to/video.mp4"
else
    python3 demo.py ctdet \
    --demo $1 \
    --load_model ./weights/ctdet_coco_dla_2x.pth
fi

