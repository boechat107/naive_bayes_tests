IMAGE_NAME=boechat107/ipython-sci
PROJDIR=/opt/naive_bayes
TARGET=$(PROJDIR)/multinomial_naive_bayes.py


docker:
	docker build -t $(IMAGE_NAME) .

run-dev:
	docker run -it --rm -v $(PWD):$(PROJDIR) $(IMAGE_NAME) bash

run:
	docker run -it --rm -v $(PWD):$(PROJDIR) $(IMAGE_NAME) ipython2 $(TARGET)

all: docker
