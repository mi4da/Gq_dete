# gunicorn_conf.py
bind: '0.0.0.0:8888' # 监听地址和端口号
workers = 2 # 进程数
worker_class = 'sync' #工作模式，可选sync, gevent, eventlet, gthread, tornado等

threads = 1 # 指定每个进程的线程数，默认为1
worker_connections = 2000 # 最大客户并发量
timeout = 60 # 超时时间，默认30s
reload = False # 开发模式，代码更新时自动重启
daemon = False # 守护Gunicorn进程，默认False

accesslog = './logs/access.log' # 访问日志文件
errorlog = './logs/error.log'
loglevel = 'debug' # 日志输出等级，debug, info, warning, error, critical
