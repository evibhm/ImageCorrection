FROM python:3.13-slim
WORKDIR /app
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx
COPY pyproject.toml .
RUN pip install .
COPY . .
EXPOSE 7860

CMD ["python", "app.py"]