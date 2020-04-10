# RTCamp 使用指南

## Web服务器 + 预编译静态文件

### Apache

MAC 自带 Apache 

	$ httpd -v
	Server version: Apache/2.4.41 (Unix)
	Server built:   Feb 29 2020 02:40:57

启动、停止和重启。因为需要占用 80 端口，会要求你输入本机的管理员密码。

	sudo apachectl start
	sudo apachectl stop
	sudo apachectl restart
	
启动后能够直接通过 [http://localhost](http://localhost) 访问，如果一切正常，能够看到 “It Works” 字样。

### 修改 Web Root

这一步也就是把 RTCamp 放到 Web Root 目录下。

Apache 的配置文件放置在 `/etc/apache2/httpd.conf`

系统自带的 Web Root 目录是：`/Library/WebServer/Documents`, 将 RTCamp 的 build.zip 包展开到这个目录下即可。
