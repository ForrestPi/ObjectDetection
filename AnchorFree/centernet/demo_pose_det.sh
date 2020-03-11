if [ $# -eq 0 ];then
    echo "No input video specificed. run as ./demo_det.sh path/to/video.mp4"
else
    python3 demo.py multi_pose \
    --demo $1 \
    --load_model ./weights/multi_pose_dla_3x.pth
fi
