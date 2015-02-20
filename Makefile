all: ga genimg catimg

# consider -march=native

ga: ga.c 
	gcc -march=native -O3 -Wall -fsigned-char -g -pthread -lrt -lm -lz -lreadline -o ga ga.c 

genimg: genimg.c
	gcc -Wall -g -o genimg genimg.c 

catimg: catimg.c
	gcc -Wall -g -o catimg catimg.c 

clean:
	rm -f ga genimg catimg
