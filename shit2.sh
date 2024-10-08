#!/bin/bash

while true; do
    # 创建一个临时 cookie jar
    cookie_jar=$(mktemp)

    # 使用 curl 发送请求
    curl -x http://127.0.0.1:7890 'https://www.allcpp.cn/allcpp/ticket/buyTicketAlipay.do?ticketTypeId=3618&count=1&nonce=fCDSGMrdw68n3sDZk2whz8dpw6YpdiEs&timeStamp=1727699430&sign=572694ef86b4e0f0964724be5cfec76c&ticketInfo=1729,1,16800&purchaserIds=1312015' \
  -H 'sec-ch-ua-platform: "Windows"' \
  -H 'Referer: https://cp.allcpp.cn/' \
  -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36' \
  -H 'Accept: application/json, text/plain, */*' \
  -H 'sec-ch-ua: "Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"' \
  -H 'Content-Type: application/json;charset=UTF-8' \
  -H 'sec-ch-ua-mobile: ?0' \
  --data-raw '{}'

    sleep 0.1
done
