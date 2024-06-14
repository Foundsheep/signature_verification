FROM pytorch/torchserve

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

RUN bash /code/create_mar.sh

CMD ["torchserve", "--start", "--ts-config", "/code/config.properties"]