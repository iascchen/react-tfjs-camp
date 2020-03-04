docker run -it --rm --name my_rtcamp -p 8000:3000 \
    -v $(pwd)/public/model:/opt/app/public/model \
    -v $(pwd)/public/data:/opt/app/public/data \
    iasc/react-tfjs-capm
