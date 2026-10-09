FROM ubuntu:24.04
RUN apt-get update && apt-get install -y openssh-server sudo && rm -rf /var/lib/apt/lists/*
RUN mkdir /run/sshd
COPY sshd_config /etc/ssh/sshd_config
COPY make_users.sh /make_users.sh
RUN chmod +x /make_users.sh
EXPOSE 22
CMD ["/make_users.sh"]
