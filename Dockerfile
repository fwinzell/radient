### Derive from the base container image
FROM dockerdex.umcn.nl:5005/diag/base-images:base-pt2.7.1

COPY . .

RUN pip3 install -r requirements.txt



