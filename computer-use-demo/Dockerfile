FROM docker.io/ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_PRIORITY=high

# ============================================
# System packages - UI, Python, and build deps
# ============================================
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install \
    # UI Requirements
    xvfb \
    xterm \
    xdotool \
    scrot \
    imagemagick \
    sudo \
    mutter \
    x11vnc \
    # Python/pyenv reqs
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    git \
    wget \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    jq \
    # Network tools
    net-tools \
    netcat-openbsd \
    gettext-base \
    # Python and essentials (FIXED: moved here)
    python3 \
    python3-pip \
    ca-certificates \
    gnupg \
    lsb-release \
    unzip \
    # PPA req
    software-properties-common && \
    # Userland apps
    add-apt-repository -y ppa:mozillateam/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libreoffice \
    firefox-esr \
    x11-apps \
    xpdf \
    gedit \
    xpaint \
    tint2 \
    galculator \
    pcmanfm && \
    # Cleanup (FIXED: at the end)
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ============================================
# noVNC for web-based VNC access
# ============================================
RUN git clone --branch v1.5.0 https://github.com/novnc/noVNC.git /opt/noVNC && \
    git clone --branch v0.12.0 https://github.com/novnc/websockify /opt/noVNC/utils/websockify && \
    ln -s /opt/noVNC/vnc.html /opt/noVNC/index.html

# ============================================
# DevOps / Infrastructure Tools
# ============================================

# AWS CLI v2
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -q awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip && \
    aws --version

# kubectl (pinned version for reproducibility)
ENV KUBECTL_VERSION=v1.31.0
RUN curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" -o kubectl && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/ && \
    kubectl version --client

# eksctl
RUN curl -fsSL "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_Linux_amd64.tar.gz" -o eksctl.tar.gz && \
    tar -xzf eksctl.tar.gz && \
    mv eksctl /usr/local/bin/ && \
    rm eksctl.tar.gz && \
    eksctl version

# Helm
RUN curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash && \
    helm version

# Docker CLI (for building images)
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    docker --version

# Terraform
ENV TERRAFORM_VERSION=1.7.5
RUN curl -fsSL "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip" -o terraform.zip && \
    unzip -q terraform.zip && \
    mv terraform /usr/local/bin/ && \
    rm terraform.zip && \
    terraform version

# ============================================
# User setup
# ============================================
ENV USERNAME=computeruse
ENV HOME=/home/$USERNAME
RUN useradd -m -s /bin/bash -d $HOME $USERNAME && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \
    # Add user to docker group (for Docker socket access if mounted)
    groupadd -f docker && \
    usermod -aG docker $USERNAME

USER computeruse
WORKDIR $HOME

# ============================================
# Python via pyenv
# ============================================
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv && \
    cd ~/.pyenv && src/configure && make -C src && cd .. && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc

ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
ENV PYENV_VERSION=3.11.6

RUN eval "$(pyenv init -)" && \
    pyenv install $PYENV_VERSION && \
    pyenv global $PYENV_VERSION && \
    pyenv rehash

# ============================================
# Python packages
# ============================================
RUN python -m pip install --upgrade pip==23.1.2 setuptools==58.0.4 wheel==0.40.0 && \
    python -m pip config set global.disable-pip-version-check true

RUN python -m pip install \
    requests \
    beautifulsoup4 \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    pillow \
    openpyxl \
    python-docx \
    PyPDF2 \
    pytest \
    black \
    flake8 \
    ipython \
    jupyter \
    scikit-learn \
    pyyaml \
    boto3 \
    psycopg2-binary \
    sqlalchemy \
    redis \
    celery \
    pyotp \
    httpx \
    # Kubernetes Python client
    kubernetes

# Application requirements
COPY --chown=$USERNAME:$USERNAME computer_use_demo/requirements.txt $HOME/computer_use_demo/requirements.txt
RUN python -m pip install -r $HOME/computer_use_demo/requirements.txt

# ============================================
# Application files
# ============================================
COPY --chown=$USERNAME:$USERNAME image/ $HOME
RUN sudo chown -R $USERNAME:$USERNAME $HOME/.cache 2>/dev/null || true
COPY --chown=$USERNAME:$USERNAME computer_use_demo/ $HOME/computer_use_demo/

# ============================================
# Display configuration
# ============================================
ARG DISPLAY_NUM=1
ARG HEIGHT=768
ARG WIDTH=1024
ENV DISPLAY_NUM=$DISPLAY_NUM
ENV HEIGHT=$HEIGHT
ENV WIDTH=$WIDTH

# ============================================
# Verify all tools are working
# ============================================
RUN echo "Verifying installations..." && \
    aws --version && \
    kubectl version --client && \
    eksctl version && \
    helm version && \
    docker --version && \
    terraform version && \
    python --version && \
    echo "All tools verified!"

ENTRYPOINT [ "./entrypoint.sh" ]