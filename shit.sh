#!/bin/bash

while true; do
    # 创建一个 cookie jar
    cookie_jar=$(mktemp)

    # 添加 cookies
    curl -s -c "$cookie_jar" -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36" \
        --data "{}" \
        -H "Content-Type: application/json;charset=UTF-8" \
        -H "Accept: application/json, text/plain, */*" \
        -H "Referer: https://cp.allcpp.cn/" \
        -H "Origin: https://cp.allcpp.cn" \
        --cookie "JALKSJFJASKDFJKALSJDFLJSF=2780154238f7eb23ed14a4456ab4b9dbeed45a322f221.12.59.212_2189247220; \
                   token=\"efgX6T3aDpHudOS5Bz/HkPuiiN5crlU3Bf/anvAWZHjBr1RdFcZPN/6w5VVc+n7l6foF2SwIyDCVidl6pSA+2WIGxICm6otigOTjI6cJU7De34FdUyux1cKIykUDpXaKPkXVQSxlr85+y2IPTJ9KC9xm+BMjO9t+dvbRRbFoaIQ=\"; \
                   Hm_lvt_75e110b2a3c6890a57de45bd2882ec7c=1727437235,1727587291,1727597654,1727666433; \
                   HMACCOUNT=39B8C18754FBA9FB; \
                   JSESSIONID=82072BA7D734AED182B1EA70911C0680; \
                   Hm_lpvt_75e110b2a3c6890a57de45bd2882ec7c=1727670014; \
                   acw_tc=27b7302317276740162712478e01f11aa51192aa57a05ffb19efd22012; \
                   cdn_sec_tc=27b7302317276740162712478e01f11aa51192aa57a05ffb19efd22012" \
        "https://www.allcpp.cn/allcpp/ticket/buyTicketAlipay.do?ticketTypeId=3629&count=1&nonce=y4i7TrkNpYSMeQ2enrkzMijZ2EJZTDjY&timeStamp=1727675217&sign=cb8d1991a407ca729788d78241c73380&ticketInfo=1729,1,9800&purchaserIds=1312015"

    sleep 0.3
done
