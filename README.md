# face-shuffle-with-openface
Extracting face from one person in the image and putting it on top of another person and vice versa. Supports multi-face images

### Set up using docker
1) *Install docker*: https://www.docker.com/products/docker

2) *Set up a docker for openface with the following steps or by going to*
https://cmusatyalab.github.io/openface/setup/

> > Automated Docker Build

> > The quickest way to getting started is to use our pre-built automated Docker build, which is available from bamos/openface. This does not require or use a locally checked out copy of OpenFace. To use on your images, share a directory between your host and the Docker container.
```sh 
$ docker pull bamos/openface
$ docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash
$ cd /root/openface
$ ./demos/compare.py images/examples/{lennon*,clapton*}
$ ./demos/classifier.py infer models/openface/celeb-classifier.nn4.small2.v1.pkl 
$ ./images/examples/carell.jpg
$ ./demos/web/start-servers.sh
````

3) *Clone or download this github project locally*

4) Run the following docker command to access the image. 
> replace <path_to_this_file> with the absolute path to getFace.py
````sh
$ docker run -v <path_to_this_file>:/mnt/host -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash
````

5) Run getFace.py
````sh
$ cd /mnt/host
$ python getFace.py
````

### Set up using source files
    Check readme-source file
