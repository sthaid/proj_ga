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

#include <stdio.h>
#include <string.h>
#include <errno.h>

#define BLACK       0
#define RED         1
#define GREEN       2
#define YELLOW      3
#define BLUE        4
#define MAGENTA     5
#define CYAN        6
#define WHITE       7

void util_print_pixel(char c);

int main(int argc, char ** argv)
{
    int img_idx, pixel_idx;
    char * fn;
    FILE * fp;
    char s[1000];

    printf("argc %d\n", argc);

    for (img_idx = 1; img_idx < argc; img_idx++) {
        fn = argv[img_idx];

        fp = fopen(fn, "r");
        if (fp == NULL) {
            printf("ERROR failed open %s, %s\n", fn, strerror(errno));
            continue;
        }

        printf("%s ...\n", fn);
        while (fgets(s, sizeof(s), fp) != NULL) {
            for (pixel_idx = 0; s[pixel_idx] != '\n' && s[pixel_idx] != '\0'; pixel_idx++) {
                util_print_pixel(s[pixel_idx]);
            }
            printf("\n");
        }
        printf("\n");

        fclose(fp);
        fp = NULL;
    }

    return 0;
}

void util_print_pixel(
    char c)
{
    int idx;

    switch (c) {
    case '.': idx = BLACK;   break;
    case 'R': idx = RED;     break;
    case 'G': idx = GREEN;   break;
    case 'Y': idx = YELLOW;  break;
    case 'B': idx = BLUE;    break;
    case 'M': idx = MAGENTA; break;
    case 'C': idx = CYAN;    break;
    case 'W': idx = WHITE;   break;
    default:  idx = -1;      break;
    }

    if (idx >= 0) {
        printf("%c[%dm  %c[39;49m", 0x1b, 100+idx, 0x1b);
    } else {
        putchar(c);
    }
}

