/*
Copyright (c) 2015 Steven Haid

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Future Work: 
// - add program arg for random number seed

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAX_X 32
#define MAX_Y 32

#define MAX_BASE_IMAGE  100000
#define MAX_VARIATION   10

#define FALSE 0
#define TRUE  1

#define MAX_COLOR   8
#define BLACK       0
#define RED         1
#define GREEN       2
#define YELLOW      3
#define BLUE        4
#define MAGENTA     5
#define CYAN        6
#define WHITE       7

typedef char (image_t)[MAX_Y][MAX_X];

image_t base_image[MAX_BASE_IMAGE];

void create_image_init(void);
void create_base_image(image_t * base_image);
void create_var_image(image_t * base_image, int image_id, int var_id, image_t * var_image);
void write_image(int image_id, int var_id, image_t * image);
int random_range(int min, int max);

// -----------------  MAIN  -------------------------------------

int main(int argc, char **argv)
{
    int i, v;
    int max_base_image = 10;
    int max_variation = 0;

    // read parans, <max_base_image> <max_variation>,
    // print usage error if input invalid
    if ((argc < 2) ||
        (argc >= 2 && sscanf(argv[1], "%d", &max_base_image) != 1) ||
        (argc >= 3 && sscanf(argv[2], "%d", &max_variation) != 1))
    {
        printf("usage: genimage <max_base_image> {<max_variation>]\n");
        exit(1);
    }

    // check range of arg values
    if (max_base_image < 1 || max_base_image > MAX_BASE_IMAGE) {
        printf("ERROR: max_base_image not in range 1..%d\n", MAX_BASE_IMAGE);
        exit(1);
    }
    if (max_variation < 0 || max_variation > MAX_VARIATION) {
        printf("ERROR: max_variation not in range 0..%d\n", MAX_VARIATION);
        exit(1);
    }

    // init the create image capability
    create_image_init();

    // print starting msg
    printf("Creating %d base images, each with %d variation images.\n",
           max_base_image, max_variation);

    // create the base images
    srandom(1);
    for (i = 0; i < max_base_image; i++) {
        create_base_image(&base_image[i]);
        write_image(i, 0, &base_image[i]);
    }

    // create the variations of the base images
    for (v = 1; v <= max_variation; v++) {
        image_t var_image;
        srandom(v+1);
        for (i = 0; i < max_base_image; i++) {
            create_var_image(&base_image[i], i, v, &var_image);
            write_image(i, v, &var_image);
        }
    }

    // return success
    return 0;
}

// -----------------  CREATE_IMAGE  -----------------------------

#define MAX_COMP 10
#define BLACK_CHAR (comp_color_char_tbl[BLACK])

typedef struct {
    image_t image;
    int   max_x;
    int   max_y;
} component_t;

static component_t comp_tbl[MAX_COMP];
static int         max_comp_tbl;

static char comp_color_char_tbl[] = {
                        '.',   // BLACK
                        'R',   // RED
                        'G',   // GREEN
                        'Y',   // YELLOW
                        'B',   // BLUE
                        'M',   // MAGENTA
                        'C',   // CYAN
                        'W'};  // WHITE

void create_image_init(void)
{
    component_t * comp; 
    int x, y;

    // horizontal rectangle
    comp = &comp_tbl[max_comp_tbl++];
    comp->max_x = MAX_X * .8;
    comp->max_y = MAX_Y * .2;
    for (y = 0; y < comp->max_y; y++) {
        for (x = 0; x < comp->max_x; x++) {
            comp->image[y][x] = TRUE;
        }
    }

    // vertical rectangle
    comp = &comp_tbl[max_comp_tbl++];
    comp->max_x = MAX_X * .2;
    comp->max_y = MAX_Y * .8;
    for (y = 0; y < comp->max_y; y++) {
        for (x = 0; x < comp->max_x; x++) {
            comp->image[y][x] = TRUE;
        }
    }

    // square
    comp = &comp_tbl[max_comp_tbl++];
    comp->max_x = MAX_X * .4;
    comp->max_y = MAX_Y * .4;
    for (y = 0; y < comp->max_y; y++) {
        for (x = 0; x < comp->max_x; x++) {
            comp->image[y][x] = TRUE;
        }
    }

    // circle
    comp = &comp_tbl[max_comp_tbl++];
    comp->max_x = MAX_X * .5;
    comp->max_y = comp->max_x;
    int r = comp->max_x / 2;
    for (y = 0; y < comp->max_y; y++) {
        for (x = 0; x < comp->max_x; x++) {
            if ((x-r)*(x-r) + (y-r)*(y-r) < r*r) {
                comp->image[y][x] = TRUE;
            }
        }
    }

    // triangle
    comp = &comp_tbl[max_comp_tbl++];
    comp->max_x = (int)(MAX_X * .7) | 1;
    int cnt = comp->max_x;
    int start = 0;
    for (y = 0; TRUE; y++) {
        for (x = start; x < start+cnt; x++) {
            comp->image[y][x] = TRUE;
        }
        if (cnt == 1) {
            comp->max_y = y + 1;
            break;
        }
        start += 1;
        cnt   -= 2;
    }
}

void create_base_image(image_t * base_image)
{
    int i, x, y;

    // create the base image
    bool      comp_used[MAX_COMP];
    bool      comp_color_used[MAX_COLOR];
    int       max_base_image_components;
    int       comp_pixels_set;

gen_base_image_again:
    comp_pixels_set = 0;
    memset(base_image, BLACK_CHAR, sizeof(image_t));
    max_base_image_components = random_range(4,4);  // max must be <= MAX_COLOR-1 and <= max_comp
    bzero(comp_used,sizeof(comp_used));
    bzero(comp_color_used,sizeof(comp_color_used));

    for (i = 0; i < max_base_image_components; i++) {
        component_t * comp;
        int           comp_idx, comp_color_idx;
        int           comp_x, comp_y;

        // determine component attributes
        while (TRUE) {
            comp_idx = random_range(0,max_comp_tbl-1);
            if (!comp_used[comp_idx]) {
                comp_used[comp_idx] = TRUE;
                comp = &comp_tbl[comp_idx];
                break;
            }
        }
        while (TRUE) {
            comp_color_idx = random_range(RED,WHITE);
            if (!comp_color_used[comp_color_idx]) {
                comp_color_used[comp_color_idx] = TRUE;
                break;
            }
        }
        comp_x = random_range(0,MAX_X-comp->max_x);
        comp_y = random_range(0,MAX_Y-comp->max_y);

        // overlay component on image
        for (y = 0; y < comp->max_y; y++) {
            for (x = 0; x < comp->max_x; x++) {
                if ((comp_y+y >= MAX_Y) || (comp_x+x >= MAX_X)) {
                    printf("ERROR: comp=%ld x,y=%d,%d comp_x,y=%d,%d comp_max_x,y=%d,%d\n",
                           comp-comp_tbl, x, y, comp_x, comp_y, comp->max_x, comp->max_y);
                    exit(1);
                }
                if (comp->image[y][x] == TRUE) {
                    (*base_image)[comp_y+y][comp_x+x] = comp_color_char_tbl[comp_color_idx];
                    comp_pixels_set++;
                }
            }
        }
    }

    // count number of pixels set in base image, 
    // if too much component overlap then try again
    int total_pixels_set = 0;
    for (y = 0; y < MAX_Y; y++) {
        for (x = 0; x < MAX_X; x++) {
            if ((*base_image)[y][x] != BLACK_CHAR) {
                total_pixels_set++;
            }
        }
    }
    if ((float)total_pixels_set/(float)comp_pixels_set < .90) {
        goto gen_base_image_again;
    }
}

void create_var_image(image_t * base_image, int image_id, int var_id, image_t * var_image)
{
    int max_random_pixel_variation, i;
    int x, y;

    // start with var_image set to base_image
    memcpy(var_image, base_image, sizeof(image_t));

    // apply translation variation
    switch ((var_id+image_id) % 4) {
    case 0:
        // shift left
        for (y = 0; y < MAX_Y; y++) {
            for (x = 0; x < MAX_X-1; x++) {
                (*var_image)[y][x] = (*var_image)[y][x+1];
            }
            (*var_image)[y][MAX_X-1] = BLACK_CHAR;
        }
        break;
    case 1:
        // shift right
        for (y = 0; y < MAX_Y; y++) {
            for (x = MAX_X-1; x > 0; x--) {
                (*var_image)[y][x] = (*var_image)[y][x-1];
            }
            (*var_image)[y][0] = BLACK_CHAR;
        }
        break;
    case 2:
        // shift up
        for (x = 0; x < MAX_X; x++) {
            for (y = MAX_Y-1; y > 0; y--) {
                (*var_image)[y][x] = (*var_image)[y-1][x];
            }
            (*var_image)[0][x] = BLACK_CHAR;
        }
        break;
    case 3:
        // shift down
        for (x = 0; x < MAX_X; x++) {
            for (y = 0; y < MAX_Y-1; y++) {
                (*var_image)[y][x] = (*var_image)[y+1][x];
            }
            (*var_image)[MAX_Y-1][x] = BLACK_CHAR;
        }
        break;
    default:
        printf("ERROR invalid var_id %d\n", var_id);
        exit(1);
    }

    // apply random pixel variation
    max_random_pixel_variation = random_range(3,8);
    for (i = 0; i < max_random_pixel_variation; i++) {
        int x = random_range(0,MAX_X-1);
        int y = random_range(0,MAX_Y-1);
        (*var_image)[y][x] = comp_color_char_tbl[random_range(BLACK,WHITE)];
    }
}

// -----------------  WRITE_IMAGE  ------------------------------

void write_image(int image_id, int var_id, image_t * image)
{
    char fn[100];
    FILE * fp;
    int x, y;

    if (var_id == 0) {
        sprintf(fn, "img%5.5d-base.img", image_id);
    } else {
        sprintf(fn, "img%5.5d-var%d.img", image_id, var_id);
    }

    fp = fopen(fn, "w");
    if (fp == NULL) {
        printf("ERROR fopen %s\n", fn);
        exit(1);
    }

    for (y = MAX_Y-1; y >= 0; y--) {
        for (x = 0; x < MAX_X; x++) {
            fputc((*image)[y][x],fp);
        }
        fputc('\n', fp);
    }

    fclose(fp);
}

// -----------------  UTILS  ------------------------------------

int random_range(int min, int max)
{
    if (min > max) {
        printf("ERROR min=%d max=%d\n", min, max);
        exit(1);
    }

    return min + ((long)(max - min + 1) * random() / ((long)RAND_MAX + 1));
}
