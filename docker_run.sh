docker run \
  --it --rm \
  --name my_rtcamp \
  -e "NODE_ENV=production" \
  -v '$pwd'/public/model /opt/app/model \
  -v '$pwd'/public/data /opt/app/data \
  -p 8000:80 \
  iasc/react-tfjs-capm
