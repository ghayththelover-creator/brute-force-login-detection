FROM ubuntu:24.04
RUN apt-get update && apt-get install -y hydra sshpass netcat-openbsd coreutils wamerican && rm -rf /var/lib/apt/lists/*
COPY attack.sh /attack.sh
COPY userlist.txt /tmp/userlist.txt
RUN chmod +x /attack.sh
CMD ["/attack.sh"]
