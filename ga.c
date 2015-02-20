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

// Ideas for future improvement:
// - support more than 128 neurons per level

//
// Description
//

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>
#include <errno.h>
#include <ctype.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <zlib.h>

//
// defines
//

// #define UNIT_TEST

#define TRUE                       1
#define FALSE                      0

#define MB                         0x100000

#define MAX_FILE_NAME              100  
#define MAX_TIMING_STAT            10
#define MAX_POP_LOG                10000000

#define MAX_ORG                    3000 
#define MAX_IMAGE                  100000
#define MAX_IMAGE_HASH_TBL         1000000
#define MAX_ANSWER                 100000
#define MAX_ANSWER_NAME            12
#define MAX_ANSWER_HASH_TBL        1000000
#define MAX_NEURON_LVL             3
#define MAX_SCORE_FBA_LOOKUP_TBL   1000000
#define MAX_X                      32
#define MAX_Y                      32
#define MAX_VERBOSE                1

#define MAX_PIXELS                 (MAX_X*MAX_Y)
#define MAX_ANSWER_NN_BITS         ((uint32_t)(sizeof(ans_nn_t) * 8))

#define MAX_COLOR                  8
#define BLACK                      0
#define RED                        1
#define GREEN                      2
#define YELLOW                     3
#define BLUE                       4
#define MAGENTA                    5
#define CYAN                       6
#define WHITE                      7

#define POP_MAGIC                  0xaabbccdd

#define SORT_KEY_TYPE_DOUBLE       1
#define SORT_KEY_TYPE_UINT32       2
#define SORT_KEY_TYPE_UINT64       3
#define SORT_KEY_TYPE_UINT128      4

#define WORK_THREAD_REQ_GIVE_TEST  1
#define WORK_THREAD_REQ_SCORE_TEST 2

#define ANSWER_IS_CORRECT          1
#define ANSWER_IS_INCORRECT        2 
#define ANSWER_IS_DONT_KNOW        3

#define POP_SIZE(_max_org,_max_chromes) \
            (sizeof(pop_t) + \
             (_max_org) * sizeof(void *) + \
             (_max_org) * (_max_chromes))

#define SCORE_SIZE(_max_org,_max_image_tbl) \
            (sizeof(score_t) + \
             (_max_org) * sizeof(void *) + \
             (_max_org) * (_max_image_tbl) * sizeof(score_org_ans_t))

#define PRINT_ERROR(fmt, args...) \
    do { \
        printf("%s: ERROR " fmt, __func__, ##args); \
    } while (0)

#define ASSERT_MSG(expr,fmt,...) \
    do { \
        if (!(expr)) { \
            util_assert(#expr, __func__, __FILE__, __LINE__, fmt, ##__VA_ARGS__); \
        } \
    } while (0)

#define GET_PARAM(_name) \
    do { \
        char s[100]; \
        printf("  %s (%d) ? ", #_name, _name); \
        fgets(s,sizeof(s),stdin); \
        sscanf(s, "%d", &_name); \
        if (ctrl_c) { \
            ctrl_c = FALSE; \
            return -1; \
        } \
    } while (0)

#define GET_PARAM_STR(_name) \
    do { \
        char s[100]; \
        printf("  %s (%s) ? ", #_name, _name); \
        fgets(s,sizeof(s),stdin); \
        sscanf(s, "%s", _name); \
        if (ctrl_c) { \
            ctrl_c = FALSE; \
            return -1; \
        } \
    } while (0)

#define PARAM_CHECK(_param, _min, _max) \
    do { \
        if ((_param) < (_min) || (_param) > (_max)) { \
            PRINT_ERROR("%s not in range %d to %d\n", #_param, (_min), (_max)); \
            return -1; \
        } \
    } while (0)

#define PAGE_ALIGN(x) ((void*)((uint64_t)(x) & 0xfffffffffffff000L))

#define MAX_ANS_NN(_max_neuron) \
            ((_max_neuron) < MAX_ANSWER_NN_BITS \
             ? ((ans_nn_t)1 << (_max_neuron)) - 1 \
             : (ans_nn_t)-1)

#define MAX_INCORRECT_NEURON_CNT (pop->param_max_incorrect_neuron_cnt[pop->neuron_lvl])

#define SETTING_VERBOSE                   (settings[0].cur_value)
#define SETTING_CROSSOVER_RATE_PERCENT    (settings[1].cur_value)
#define SETTING_LEARN_TERM_ORG_SCORE_AVG  (settings[2].cur_value)
#define SETTING_LEARN_TERM_GEN_COUNT      (settings[3].cur_value)
#define SETTING_TEST_MAX_ORG              (settings[4].cur_value)
#define MAX_SETTINGS (sizeof(settings) / sizeof(settings[0]))
#define SETTINGS_TABLE \
            { "verbose",                   0,       1,   0 }, \
            { "crossover_rate_percent",    0,     100,  70 }, \
            { "learn_term_org_score_avg", 60,     100,  95 }, \
            { "learn_term_gen_count",      1, 1000000,  10 }, \
            { "test_max_org",              0, 1000000,   0}, \

//
// typedefs
//

typedef __uint128_t ans_nn_t;

typedef struct {
    uint64_t total;
    uint64_t last;
    uint64_t stat[MAX_TIMING_STAT];
} timing_t;

typedef struct {
    char filename[MAX_FILE_NAME];
    char ans_name[MAX_ANSWER_NAME];
    char pixels[MAX_PIXELS];
} image_t;

typedef struct {
    uint64_t sort_key;
    double   score;
    uint32_t idx;
} score_sort_t;

#pragma pack(push,1)
typedef struct {
    uint8_t  ans_status;
    uint8_t  incorrect_neuron_count;
    ans_nn_t ans_nn;
} score_org_ans_t;
#pragma pack(pop)

#pragma pack(push,1)
typedef struct {
    uint32_t state;
    ans_nn_t ans_nn;
    uint32_t best_ans;
    uint16_t best_ans_incorrect_neuron_cnt;
} score_fba_lookup_ent_t;
#pragma pack(pop)

typedef struct {
    image_t              * image_tbl;
    uint32_t               max_image_tbl;
    uint32_t               max_org; 

    double                 org_score[MAX_ORG];
    score_sort_t           org_score_sort[MAX_ORG];
    double                 org_score_max;
    double                 org_score_avg;
    double                 org_score_min;

    double                 org_weighted_score[MAX_ORG];
    score_sort_t           org_weighted_score_sort[MAX_ORG];
    double                 org_weighted_score_max;
    double                 org_weighted_score_avg;
    double                 org_weighted_score_min;

    score_fba_lookup_ent_t fba_lookup_tbl[MAX_SCORE_FBA_LOOKUP_TBL];

    score_org_ans_t      * org_answer[];
} score_t;

typedef struct {
    uint32_t     magic;
    size_t       pop_size;

    uint32_t     param_max_org;
    uint32_t     param_max_neuron[MAX_NEURON_LVL];
    uint32_t     param_max_incorrect_neuron_cnt[MAX_NEURON_LVL];
    char         param_image_filename[100];

    uint32_t     gen_num;

    double       org_score_max;
    double       org_score_avg;
    double       org_score_min;
    double       org_weighted_score_max;
    double       org_weighted_score_avg;
    double       org_weighted_score_min;

    char         log[MAX_POP_LOG];
    uint32_t     max_log;

    uint32_t     max_answer;
    struct {
        ans_nn_t nn;
        char     name[MAX_ANSWER_NAME];
        uint32_t image_tbl_idx;
    } answer[MAX_ANSWER];
    uint32_t     answer_hash_tbl[MAX_ANSWER_HASH_TBL];

    uint32_t     max_image_tbl;
    image_t      image_tbl[MAX_IMAGE];
    uint32_t     image_to_ans[MAX_IMAGE];  
    bool         image_is_primary[MAX_IMAGE];
    uint32_t     image_hash_tbl[MAX_IMAGE_HASH_TBL];

    int32_t      neuron_lvl;
    uint32_t     max_chromes;
    uint32_t     max_chromes_lvl[MAX_NEURON_LVL];
    char       * org[];
} pop_t;

typedef struct {
    uint32_t req;
    union {
        struct {
            uint32_t   org_idx;
        } give_test;
        struct {
            uint32_t   org_idx;
        } score_test;
    } u;
} work_thread_req_t;

typedef struct {
    char * name;
    int32_t min_value;
    int32_t max_value;
    int32_t cur_value;
} settings_t;

//
// variables
//

pop_t           * pop;
score_t         * score;
work_thread_req_t wt_req;
pthread_barrier_t wt_barrier;
char              pop_filename[MAX_FILE_NAME];
bool              ctrl_c;
settings_t        settings[] = { SETTINGS_TABLE };

//
// prototypes
//

// main routines
int32_t main(
    int32_t argc, 
    char **argv);
int32_t init(
    void);

// command routines
uint32_t cmd_processor(
    char * script_filename);
int32_t cmd_create(
    char * args);
int32_t cmd_read(
    char * args);
int32_t cmd_write(
    char * args);
int32_t cmd_learn(
    char * args);
int32_t cmd_test(
    char * args);
int32_t cmd_display(
    char * args);
int32_t cmd_script(
    char * args);
int32_t cmd_set(
    char * args);
int32_t cmd_shell(
    char * args);
int32_t cmd_remark(
    char * args);

// ga routines
int32_t ga_learn(
    uint32_t neuron_lvl);
void ga_learn_choose_best_answers(
    void);
void ga_learn_sort_orgs(
    void);
void ga_learn_create_next_gen(
    void);
void ga_learn_create_children(
    char * parent1, 
    char * parent2, 
    char * child1, 
    char * child2);
int32_t ga_test(
    image_t * image_tbl, 
    uint32_t max_image_tbl,
    uint32_t max_org);
char * ga_test_score_org_ans_info_str(
    uint32_t org_idx,
    uint32_t ans_status,
    char * s);
int32_t ga_test_init(
    image_t * image_tbl,
    uint32_t max_image_tbl,
    uint32_t max_org);
void ga_test_complete(
    void);
int32_t ga_test_give(
    void);
int32_t ga_test_score(
    void);
void ga_test_score_find_best_answer(
    ans_nn_t ans_nn,
    uint32_t * best_ans_arg,
    uint32_t * best_ans_incorrect_neuron_cnt_arg);
void ga_test_score_compute_stats(
    double * values,
    uint32_t max_values,
    double * max_arg, 
    double * avg_arg, 
    double * min_arg);
void ga_display(
    void);

// neural net routines
ans_nn_t nn_eval(
    uint32_t org_idx, 
    char * pixels);

// util routines
int32_t util_write_pop_file(
    char * pop_fn);
int32_t util_read_pop_file(
    char * pop_fn);
int32_t util_read_images_from_filename(
        char * filename_list, 
        image_t ** image_tbl_ret, 
        uint32_t * max_image_tbl_ret);
int32_t util_read_image_file(
    char * image_fn, 
    image_t * image);
void util_print_pixel(
    uint32_t color);
uint32_t util_random(
    uint32_t max_val);
void util_random_set_default_seed(
    void);
void util_plot_histogram(
    char * hist_name_str,
    double min_x,
    double max_x,
    uint32_t max_data,
    double * data);
void util_plot_histogram_org_score(
    void);
void util_plot_histogram_org_weighted_score(
    void);
void util_plot_histogram_ans_nn(
    void);
void util_sort(
    void * array,
    uint32_t elements,
    uint32_t element_size,
    off_t sort_key_offset,
    uint32_t sort_key_type);
ans_nn_t util_choose_best_ans_nn(
    uint32_t img_idx);
void util_thread_create(
    void * (*proc)(void * cx),
    void * cx);
void * util_work_thread(
    void * cx);
uint32_t util_hash_lookup_ans_name(
    char * ans_name);
void util_hash_add_ans_name(
    char * ans_name,
    uint32_t ans);
uint32_t util_hash_lookup_image(
    image_t * image);
void util_hash_add_image(
    image_t * image,
    uint32_t img);
void util_ctrl_c_init(
    void);
void util_ctrl_c_hndlr(
    int32_t sig);
void util_assert(
    const char *expression, 
    const char *func, 
    const char *file,
    uint32_t line, 
    char * fmt, ...) __attribute__ ((format (printf, 5, 6)));
void util_timing(
    timing_t * tmg,
    uint32_t idx);
uint64_t util_get_time_ns(
    void);
void util_pop_log(
    char * fmt, ...) __attribute__ ((format (printf, 1, 2)));
char * util_ans_nn_to_str(
    ans_nn_t val,
    uint32_t base);
void util_parse_input_line(
    char * input_line,
    char ** cmd,
    char ** args);
uint32_t num_bits_diff(
    __uint128_t a, 
    __uint128_t b);
uint32_t util_popcount_x86(
    uint64_t x);
uint32_t util_popcount_wikipedia(
    uint64_t x);
void util_swap(
    void * x, 
    void * y, 
    size_t len);

// -----------------  USER INTFC COMMANDS  ---------------------------

int32_t main(int32_t argc, char **argv)
{
    int32_t ret;
    struct rlimit    rl;

    // set resource limti to allow core dumps
    rl.rlim_cur = RLIM_INFINITY;
    rl.rlim_max = RLIM_INFINITY;
    setrlimit(RLIMIT_CORE, &rl);

    // initialize
    ret = init();
    ASSERT_MSG(ret == 0, NULL);

    // invoke the command processor
    cmd_processor("");
    return 0;
}

int32_t init(
    void)
{
    uint64_t i;
    uint32_t num_proc;

    // set stdout to unbufferred
    setbuf(stdout, NULL);

    // register for ctrl c
    util_ctrl_c_init();

    // disable readline filename completion
    rl_bind_key('\t',rl_abort);

    // create worker threads
    num_proc = sysconf(_SC_NPROCESSORS_ONLN);
    ASSERT_MSG(num_proc >= 1, "num_proc %d", num_proc);
    pthread_barrier_init(&wt_barrier,NULL,num_proc+1);
    for (i = 0; i < num_proc; i++) {
        util_thread_create(util_work_thread,(void*)i);
    }

    // return success
    return 0;
}

uint32_t cmd_processor(
    char * script_filename)
{
    #define MAX_CMD_TBL (sizeof(cmd_tbl)/sizeof(cmd_tbl[0]))

    static struct {
        char * cmd;
        int32_t (*proc)(char*);
    } cmd_tbl[] = {
        { "create",        cmd_create      },
        { "read",          cmd_read        },
        { "write",         cmd_write       },
        { "learn",         cmd_learn       },
        { "test",          cmd_test        },
        { "display",       cmd_display     },
        { "script",        cmd_script      },
        { "set",           cmd_set         },
        { "shell",         cmd_shell       },
        { "remark",        cmd_remark      },
                             };

    int32_t ret = 0;
    FILE * fp;

    // set fp from which commands will be read;
    // if script_filename is provided then open it,
    // otherwise use stdin
    if (script_filename[0] != '\0') {
        fp = fopen(script_filename, "r");
        if (fp == NULL) {
            PRINT_ERROR("open '%s'\n", script_filename);
            return -1;
        }
    } else {
        fp = stdin;
    }

    // command processing loop
    while (TRUE) {
        int32_t  i;
        char   * cmd;
        char   * args;
        char     cmd_line[10000];
        char     prompt[1000];

        // generate prompt string
        sprintf(prompt, "%s%s%s%s> ", 
                script_filename[0] != '\0' ? " " : "",
                script_filename,
                pop_filename[0] != '\0' ? " " : "",
                pop_filename);

        // read cmdline
        if (fp == stdin) {
            char * rl;

            rl = readline(prompt);
            if (rl == NULL) {
                break;
            }
            if (rl[0] != '\0') {
                add_history(rl);
            }
            strcpy(cmd_line, rl);
            free(rl);
        } else {
            if (fgets(cmd_line, sizeof(cmd_line), fp) == NULL) {
                break;
            }
            printf("%s%s", prompt, cmd_line);
        }

        // parse to cmd / args, continue if blank line
        util_parse_input_line(cmd_line, &cmd, &args);

        // check for empty or comment command
        if (strcmp(cmd, "") == 0 || strcmp(cmd, "#") == 0) {
            continue;
        }

        // check for quit command
        if (strcmp(cmd, "q") == 0) {
            break;
        }

        // search cmd_tbl for matching cmd; 
        // if found then call cmd procedure else error
        for (ret = 0, i = 0; i < MAX_CMD_TBL; i++) {
            if (strcmp(cmd, cmd_tbl[i].cmd) == 0) {
                if (cmd_tbl[i].proc(args) < 0) {
                    PRINT_ERROR("processing '%s'\n", cmd);
                    ret = -1;
                }
                break;
            }
        }
        if (i == MAX_CMD_TBL) {
            PRINT_ERROR("invalid cmd '%s'\n", cmd);
            ret = -1;
        }

        // if an error occurred and we're running from a script cmd then break
        // out of the cmd processing loop; else reset ret to 0
        if (fp != stdin && ret != 0) {
            break;
        }
        ret = 0;
    }

    // if processing a script file then close it and print terminating
    if (fp != stdin) {
        fclose(fp);
        printf("script %s terminating\n", script_filename);
    }

    // return 
    return ret;
}

int32_t cmd_create(
    char * args)
{
    uint32_t   param_max_org;
    uint32_t   param_max_neuron_lvl0;
    uint32_t   param_max_neuron_lvl1;
    uint32_t   param_max_neuron_lvl2;
    uint32_t   param_max_incorrect_neuron_cnt_lvl0;
    uint32_t   param_max_incorrect_neuron_cnt_lvl1;
    uint32_t   param_max_incorrect_neuron_cnt_lvl2;
    char       param_image_filename[100];
    int32_t    cnt;
    size_t     pop_size;
    uint32_t   org_idx, chrome_idx;
    uint32_t   max_chromes;
    uint32_t   max_chromes_lvl0;
    uint32_t   max_chromes_lvl1;
    uint32_t   max_chromes_lvl2;

    // args:
    // - param_max_org
    // - param_max_neuron_lvl0
    // - param_max_neuron_lvl1
    // - param_max_neuron_lvl2
    // - param_max_incorrect_neuron_cnt_lvl0;
    // - param_max_incorrect_neuron_cnt_lvl1;
    // - param_max_incorrect_neuron_cnt_lvl2;
    // - param_image_filename

    // set default param values
    param_max_org                       = 500;
    param_max_neuron_lvl0               = 128;
    param_max_neuron_lvl1               = 30;
    param_max_neuron_lvl2               = 0;
    param_max_incorrect_neuron_cnt_lvl0 = 30;
    param_max_incorrect_neuron_cnt_lvl1 = 8;
    param_max_incorrect_neuron_cnt_lvl2 = 0;
    strcpy(param_image_filename, "img/*base.img");

    // get param values from user input
    cnt = sscanf(args, "%d %d %d %d %d %d %d %s",
                 &param_max_org,
                 &param_max_neuron_lvl0,
                 &param_max_neuron_lvl1,
                 &param_max_neuron_lvl2,
                 &param_max_incorrect_neuron_cnt_lvl0,
                 &param_max_incorrect_neuron_cnt_lvl1,
                 &param_max_incorrect_neuron_cnt_lvl2,
                 param_image_filename);
    if (cnt != 8) {
        GET_PARAM(param_max_org);
        GET_PARAM(param_max_neuron_lvl0);
        GET_PARAM(param_max_neuron_lvl1);
        GET_PARAM(param_max_neuron_lvl2);
        GET_PARAM(param_max_incorrect_neuron_cnt_lvl0);
        GET_PARAM(param_max_incorrect_neuron_cnt_lvl1);
        GET_PARAM(param_max_incorrect_neuron_cnt_lvl2);
        GET_PARAM_STR(param_image_filename);
    }

    // reseed random number with default seed
    util_random_set_default_seed();

    // determine max values
    max_chromes_lvl0 = param_max_neuron_lvl0 * (MAX_PIXELS * MAX_COLOR * sizeof(char));
    max_chromes_lvl1 = param_max_neuron_lvl1 * (param_max_neuron_lvl0 * sizeof(char));
    max_chromes_lvl2 = param_max_neuron_lvl2 * (param_max_neuron_lvl1 * sizeof(char));
    max_chromes = max_chromes_lvl0 + max_chromes_lvl1 + max_chromes_lvl2;  

    // print size of data structs
    printf("sizeof pop_t   %ld MB\n", POP_SIZE(param_max_org,max_chromes) / MB);
#if 0
    printf("   max_chromes %d (%d + %d + %d)\n", 
           max_chromes, max_chromes_lvl0, max_chromes_lvl1, max_chromes_lvl2);
    printf("   answer      %ld MB\n", sizeof(pop->answer)/MB);
    printf("   log         %ld MB\n", sizeof(pop->log)/MB);
    printf("   image_tbl   %ld MB\n", sizeof(pop->image_tbl)/MB);
    printf("   org         %ld MB\n", (uint64_t)max_chromes * param_max_org/ MB);
#endif
    printf("sizeof score_t %ld MB (for max of %d imgaes)\n", SCORE_SIZE(param_max_org, MAX_IMAGE) / MB, MAX_IMAGE);

    // verify params, try to use realistic ranges
    PARAM_CHECK(param_max_org, 20, MAX_ORG);

    PARAM_CHECK(param_max_neuron_lvl0, 20, MAX_ANSWER_NN_BITS);
    PARAM_CHECK(param_max_incorrect_neuron_cnt_lvl0, 0, MAX_ANSWER_NN_BITS/3);

    if (param_max_neuron_lvl1 == 0) {
        PARAM_CHECK(param_max_neuron_lvl1, 0, 0);
        PARAM_CHECK(param_max_incorrect_neuron_cnt_lvl1, 0, 0);
        PARAM_CHECK(param_max_neuron_lvl2, 0, 0);
    } else {
        PARAM_CHECK(param_max_neuron_lvl1, 5, MAX_ANSWER_NN_BITS);
        PARAM_CHECK(param_max_incorrect_neuron_cnt_lvl1, 0, MAX_ANSWER_NN_BITS/3);
    }

    if (param_max_neuron_lvl2 == 0) {
        PARAM_CHECK(param_max_neuron_lvl2, 0, 0);
        PARAM_CHECK(param_max_incorrect_neuron_cnt_lvl2, 0, 0);
    } else {
        PARAM_CHECK(param_max_neuron_lvl2, 5, MAX_ANSWER_NN_BITS);
        PARAM_CHECK(param_max_incorrect_neuron_cnt_lvl2, 0, MAX_ANSWER_NN_BITS/3);
    }

    // additional param checks
    if (param_max_org & 1) {
        PRINT_ERROR("param_max_org%d must be even\n", param_max_org);
        return -1;
    }

    // free pop allocations, and clear pop_filename
    free(pop);
    pop = NULL;
    pop_filename[0] = '\0';

    // allocate pop
    pop_size = POP_SIZE(param_max_org,max_chromes);
    pop = malloc(pop_size);
    if (pop == NULL) {
        PRINT_ERROR("alloc pop, size %ld MB\n", pop_size/MB);
        return -1;
    }
    bzero(pop, pop_size);

    // init non-zero fields of pop
    pop->param_max_org                     = param_max_org;
    pop->param_max_neuron[0]               = param_max_neuron_lvl0;
    pop->param_max_neuron[1]               = param_max_neuron_lvl1;
    pop->param_max_neuron[2]               = param_max_neuron_lvl2;
    pop->param_max_incorrect_neuron_cnt[0] = param_max_incorrect_neuron_cnt_lvl0;
    pop->param_max_incorrect_neuron_cnt[1] = param_max_incorrect_neuron_cnt_lvl1;
    pop->param_max_incorrect_neuron_cnt[2] = param_max_incorrect_neuron_cnt_lvl2;
    strcpy(pop->param_image_filename, param_image_filename);

    pop->magic              = POP_MAGIC;
    pop->pop_size           = pop_size;
    pop->neuron_lvl         = -1;
    pop->max_chromes        = max_chromes;
    pop->max_chromes_lvl[0] = max_chromes_lvl0;
    pop->max_chromes_lvl[1] = max_chromes_lvl1;
    pop->max_chromes_lvl[2] = max_chromes_lvl2;

    memset(pop->answer_hash_tbl, 0xff, sizeof(pop->answer_hash_tbl));
    memset(pop->image_hash_tbl, 0xff, sizeof(pop->image_hash_tbl));

    for (org_idx = 0; org_idx < param_max_org; org_idx++) {
        pop->org[org_idx] = (void*)pop + sizeof(pop_t) + 
                            param_max_org* sizeof(void *) +
                            org_idx * max_chromes;
    }

    for (org_idx = 0; org_idx < param_max_org; org_idx++) {
        uint16_t * org_uint16 = (uint16_t*)pop->org[org_idx];
        for (chrome_idx = 0; chrome_idx < max_chromes/2; chrome_idx++) {
            *org_uint16 = random();
            org_uint16++;
        }
        pop->org[org_idx][max_chromes-1] = random();
    }

    // read the images and add the images to pop
    uint32_t images_with_new_ans_name_added         = 0;
    uint32_t images_with_existing_ans_name_added    = 0;
    uint32_t images_skipped_because_already_in_pop  = 0;
    image_t  * image_tbl;
    uint32_t   max_image_tbl;
    uint32_t   image_tbl_idx;

    // read the images
    if (util_read_images_from_filename(param_image_filename, &image_tbl, &max_image_tbl) < 0) {
        PRINT_ERROR("reading images from '%s'\n", param_image_filename);
        return -1;
    }

    // add the images to pop
    for (image_tbl_idx = 0; image_tbl_idx < max_image_tbl; image_tbl_idx++) {
        image_t * image = &image_tbl[image_tbl_idx];
        uint32_t  ans;

        // if this image is already in pop then skip it
        if (util_hash_lookup_image(image) != -1) {
            images_skipped_because_already_in_pop++;
            continue;
        }

        // verify image table is not full
        if (pop->max_image_tbl >= MAX_IMAGE) {
            PRINT_ERROR("image table is full\n");
            return -1;
        }

        // check if the image answer name is not already in pop; 
        if ((ans = util_hash_lookup_ans_name(image->ans_name)) == -1) {
            // verify answer table is not full
            if (pop->max_answer >= MAX_ANSWER) {
                PRINT_ERROR("answer values have all been used\n");
                return -1;
            }

            // add entry to answer table,
            strcpy(pop->answer[pop->max_answer].name, image->ans_name); 
            pop->answer[pop->max_answer].nn = 0;  // filled in by ga_learn()
            pop->answer[pop->max_answer].image_tbl_idx = pop->max_image_tbl;
            util_hash_add_ans_name(image->ans_name, pop->max_answer);

            // add entry to image table
            pop->image_tbl[pop->max_image_tbl] = *image;
            pop->image_to_ans[pop->max_image_tbl] = pop->max_answer;
            pop->image_is_primary[pop->max_image_tbl] = TRUE;
            util_hash_add_image(&pop->image_tbl[pop->max_image_tbl], pop->max_image_tbl);

            // increment max table counts
            pop->max_image_tbl++;
            pop->max_answer++;

            // increment counter
            images_with_new_ans_name_added++;
        } else {
            // add entry to image table
            pop->image_tbl[pop->max_image_tbl] = *image;
            pop->image_to_ans[pop->max_image_tbl] = ans;
            pop->image_is_primary[pop->max_image_tbl] = FALSE;
            util_hash_add_image(&pop->image_tbl[pop->max_image_tbl], pop->max_image_tbl);
            pop->max_image_tbl++;

            // increment counter
            images_with_existing_ans_name_added++;
        }
    }
    printf("completed adding images to pop: new=%d existing=%d skipped=%d\n",
           images_with_new_ans_name_added,
           images_with_existing_ans_name_added,
           images_skipped_because_already_in_pop);

    // verify that we have images 
    if (pop->max_image_tbl == 0) {
        PRINT_ERROR("no images to learn\n");
        return -1;
    }

    // add pop log entry         
    util_pop_log("create successful");

    // return success
    return 0;
}

int32_t cmd_read(
    char * args)
{
    char * pop_fn = args;

    // args:
    // - pop filename

    // reseed random number with default seed
    util_random_set_default_seed();

    // verify pop filename arg is supplied
    if (pop_fn[0] == '\0' ) {
        PRINT_ERROR("pop filename arg is required\n");
        return -1;
    }

    // read the pop file
    if (util_read_pop_file(pop_fn) < 0) {
        PRINT_ERROR("read pop file '%s'\n", pop_fn);
        return -1;
    }

    // return success
    return 0;
}

int32_t cmd_write(
    char * args)
{
    char  * pop_fn = args;

    // args:
    // - pop filename

    // verify population has been loaded
    if (pop == NULL) {
        PRINT_ERROR("population not loaded\n");
        return -1;
    }

    // verify pop filename arg is supplied
    if (pop_fn[0] == '\0' ) {
        PRINT_ERROR("pop filename arg is required\n");
        return -1;
    }

    // write the file
    if (util_write_pop_file(pop_fn) < 0) {
        PRINT_ERROR("write pop file %s\n", pop_fn);
        return -1;
    }

    // return success
    return 0;
}

int32_t cmd_learn(
    char * args)
{
    uint32_t neuron_lvl = 0;

    // args:
    // - neuron level

    // verify population has been loaded
    if (pop == NULL) {
        PRINT_ERROR("population not loaded\n");
        return -1;
    }

    // parse the args
    if (sscanf(args, "%d", &neuron_lvl) != 1) {
        PRINT_ERROR("neuron_lvl expected\n");
        return -1;
    }
    if ((neuron_lvl < 0) || 
        (neuron_lvl >= MAX_NEURON_LVL) ||
        (pop->param_max_neuron[neuron_lvl] == 0) ||
        (neuron_lvl >= 1 && pop->param_max_neuron[neuron_lvl-1] == 0) ||
        (neuron_lvl >= 2 && pop->param_max_neuron[neuron_lvl-2] == 0) ||
        (neuron_lvl != pop->neuron_lvl && neuron_lvl != pop->neuron_lvl + 1)) 
    {
        PRINT_ERROR("invalid neuron_lvl %d, pop->neuron_lvl=%d\n", 
                    neuron_lvl, pop->neuron_lvl);
        return -1;
    }

    // call ga_learn 
    if (ga_learn(neuron_lvl) < 0) {
        PRINT_ERROR("ga_learn failed\n");
        return -1;
    }

    // return success
    return 0;
}

int32_t cmd_test(
    char * args)
{
    char    * image_filename = args;
    image_t * image_tbl;
    uint32_t  max_image_tbl;
    uint32_t  max_org;

    // args
    // - image_filename   OPTIONAL

    // if image_filename is not supplied then the learned images,
    // which are stored in pop, are used

    // verify population has been loaded
    if (pop == NULL) {
        PRINT_ERROR("population not loaded\n");
        return -1;
    }

    // if no filename_list then 
    //   test using the images in pop
    // else
    //   read the test images from the filename_list
    // endif
    if (image_filename[0] == '\0') {
        image_tbl = pop->image_tbl;
        max_image_tbl = pop->max_image_tbl;
    } else  if (util_read_images_from_filename(image_filename, &image_tbl, &max_image_tbl) < 0) {
        PRINT_ERROR("reading filename_list\n");
        return -1;
    }

    // call ga_test 
    max_org = (SETTING_TEST_MAX_ORG <= 0 || SETTING_TEST_MAX_ORG > pop->param_max_org
               ? pop->param_max_org
               : SETTING_TEST_MAX_ORG);
    if (ga_test(image_tbl, max_image_tbl, max_org) < 0) {
        PRINT_ERROR("ga_test failed\n");
        return -1;
    }

    // return success
    return 0;
}

int32_t cmd_display(
    char * args)
{
    // args
    // - none

    // verify population has been loaded
    if (pop == NULL) {
        PRINT_ERROR("population not loaded\n");
        return -1;
    }
 
    // call ga_display 
    ga_display();

    // return success
    return 0;
}

int32_t cmd_script(
    char * args)
{
    char   * script_fn = args;
    int32_t  ret;

    // args
    // - script filename

    // verify script filename arg is supplied
    if (script_fn[0] == '\0' ) {
        PRINT_ERROR("script filename arg is required\n");
        return -1;
    }

    // call cmd_processor
    ret = cmd_processor(script_fn);

    // return status
    return ret;
}

int32_t cmd_set(
    char * args)
{
    int i;

    // args:
    // - name (OPTIONAL)
    // - value (OPTIONAL)

    if (args[0] == '\0') {
        for (i = 0; i < MAX_SETTINGS; i++) {
            printf("%s = %d\n", settings[i].name, settings[i].cur_value);
        }
        return 0;
    } else {
        char  name[100];
        int32_t value;

        if (sscanf(args, "%s %d", name, &value) != 2) {
            PRINT_ERROR("invalid input\n");
            return -1;
        }
        for (i = 0; i < MAX_SETTINGS; i++) {
            if (strcmp(settings[i].name, name) == 0) {
                if (value >= settings[i].min_value && value <= settings[i].max_value) {
                    settings[i].cur_value = value;
                    return 0;
                } else {
                    PRINT_ERROR("value %d not in range %d - %d\n",
                                value, settings[i].min_value, settings[i].max_value);
                    return -1;
                }
            }
        }
        PRINT_ERROR("setting '%s' does not exist\n", name);
        return -1;
    }
}

int32_t cmd_shell(
    char * args)
{
    int32_t ret;

    ret = system(args);
    return ret;
}

int32_t cmd_remark(
    char * args)
{
    // do nothing
    return 0;
}

// -----------------  GA LEARN  --------------------------------------

int32_t ga_learn(
    uint32_t neuron_lvl)
{
    uint32_t gen_count      = 0;
    int32_t  ret            = 0;
    char     term_str[100]  = "????";
    char     benefit_str[100];
    timing_t timing;

    #define TIMING_INIT                    -1
    #define TIMING_IGNORE                  -2
    #define TIMING_TEST_GIVE               0
    #define TIMING_TEST_SCORE              1
    #define TIMING_MODIFY_ANSWERS          2
    #define TIMING_DISCARD_LOW_SCORE_ORGS  3
    #define TIMING_CREATE_NEXT_GEN         4

    // if this is the first call training this neuron level then
    // - init the pop->answer[].nn to random values
    // - set all org prior neuron_lvl chromes using the chromes from 
    //   the top scoring org at the prior_neuron_lvl
    // - set the new depth of the neural net
    // endif
    if (neuron_lvl == pop->neuron_lvl + 1) {
        int i;

        printf("Initializing learn for %d neuron levels\n", neuron_lvl);

        // init the pop->answer[].nn to undefined value0
        for (i = 0; i < pop->max_answer; i++) {
            pop->answer[i].nn = -1;
        }

        // set all org prior neuron_lvl chromes using the chromes from 
        // the top scoring org at the prior_neuron_lvl
        if (neuron_lvl == 1) {
            for (i = 1; i < pop->param_max_org; i++) {
                memcpy(pop->org[i], 
                       pop->org[0], 
                       pop->max_chromes_lvl[0]);
            }
        }
        if (neuron_lvl == 2) {
            for (i = 1; i < pop->param_max_org; i++) {
                memcpy(pop->org[i] + pop->max_chromes_lvl[0], 
                       pop->org[0] + pop->max_chromes_lvl[0], 
                       pop->max_chromes_lvl[1]);
            }
        }

        // set the new depth of the neural net, 
        // reset gen_num to zero
        pop->neuron_lvl = neuron_lvl;
        pop->gen_num = 0;
    }

    // call ga_test_init
    if (ga_test_init(pop->image_tbl, pop->max_image_tbl, pop->param_max_org) == -1) {
        PRINT_ERROR("ga_test_init failed\n");
        return -1;
    }
    printf("\n");

    // log learn startig message
    util_pop_log("learn starting - "
                 "neuron_lvl %d, term_org_score_avg %d, term_gen_count %d",
                 neuron_lvl,
                 SETTING_LEARN_TERM_ORG_SCORE_AVG, 
                 SETTING_LEARN_TERM_GEN_COUNT);

    // learn loop
    util_timing(&timing, TIMING_INIT);
    while (TRUE) {
        // print header
        printf("\n------------------------- GEN NUM / COUNT : %d / %d -------------------------\n\n",   
               pop->gen_num, gen_count);

        // call ga_test_give to have all org evaluate all images that are stored in pop
        if ((ret = ga_test_give()) == -1) {
            printf(" - interrupted\n");
            sprintf(term_str, "interrupted");
            break;
        }
        util_timing(&timing, TIMING_TEST_GIVE);

        // call ga_test_score to compute test score
        if ((ret = ga_test_score()) == -1) {
            printf(" - interrupted\n");
            sprintf(term_str, "interrupted");
            break;
        }
        util_timing(&timing, TIMING_TEST_SCORE);

        // if generation number is less than 5 then
        //   . use ga_learn_choose_best_answers to improve score, and
        //   . rescore
        //   . set benefit_str
        // endif
        benefit_str[0] = '\0';
        if (pop->gen_num < 5) {
            double   orig_org_score_avg;
            double   new_org_score_avg;

            printf(" - choose_best_answers");
            ga_learn_choose_best_answers();
            util_timing(&timing, TIMING_MODIFY_ANSWERS);

            orig_org_score_avg = score->org_score_avg;
            if ((ret = ga_test_score()) == -1) {
                printf(" - interrupted\n");
                sprintf(term_str, "interrupted");
                break;
            }
            util_timing(&timing, TIMING_TEST_SCORE);
            new_org_score_avg = score->org_score_avg;

            sprintf(benefit_str, "- choose_best_ans_ben %0.1f",
                    new_org_score_avg - orig_org_score_avg);
        }

        // print done
        printf(" - done\n\n");

        // save current org_score in pop
        pop->org_score_max          = score->org_score_max;
        pop->org_score_avg          = score->org_score_avg;
        pop->org_score_min          = score->org_score_min;
        pop->org_weighted_score_max = score->org_weighted_score_max;
        pop->org_weighted_score_avg = score->org_weighted_score_avg;
        pop->org_weighted_score_min = score->org_weighted_score_min;

        // print and log progress info
        util_pop_log(
            "os %d %d %d ws %d %d %d %s",
            (uint32_t)score->org_score_max,
            (uint32_t)score->org_score_avg,
            (uint32_t)score->org_score_min,
            (uint32_t)score->org_weighted_score_max,
            (uint32_t)score->org_weighted_score_avg,
            (uint32_t)score->org_weighted_score_min,
            benefit_str);
        printf("\n");
        util_plot_histogram_org_score();
        util_plot_histogram_org_weighted_score();

        // if a terminate condition is met
        if (score->org_score_avg >= SETTING_LEARN_TERM_ORG_SCORE_AVG) {
            sprintf(term_str, "org_score_avg %0.1f >= %d",
                    score->org_score_avg, SETTING_LEARN_TERM_ORG_SCORE_AVG);
            break;
        }
        if (gen_count >= SETTING_LEARN_TERM_GEN_COUNT) {
            sprintf(term_str, "gen_count %d >= %d",
                    gen_count, SETTING_LEARN_TERM_GEN_COUNT);
            break;
        }
        util_timing(&timing, TIMING_IGNORE);

        // call ga_learn_create_next_gen to create the next generation, and
        // increment generation number and count
        ga_learn_create_next_gen();
        util_timing(&timing, TIMING_CREATE_NEXT_GEN);
        pop->gen_num++;
        gen_count++;
    }

    // sort orgs so that when a subsequent test is run on the pop, the test
    // can select to test just the first few orgs and it will be testing the
    // highest scorers
    if (strcmp(term_str, "interrupted") != 0) {
        ga_learn_sort_orgs();
    }

    // log completeion
    util_pop_log("learn completed - %s - time %ds %d %d %d %d %d", 
                 term_str,
                 (uint32_t)(timing.total / 1000000000),
                 (uint32_t)(100.0 * timing.stat[TIMING_TEST_GIVE] / timing.total + 0.5),
                 (uint32_t)(100.0 * timing.stat[TIMING_TEST_SCORE] / timing.total + 0.5),
                 (uint32_t)(100.0 * timing.stat[TIMING_MODIFY_ANSWERS] / timing.total + 0.5),
                 (uint32_t)(100.0 * timing.stat[TIMING_DISCARD_LOW_SCORE_ORGS] / timing.total + 0.5),
                 (uint32_t)(100.0 * timing.stat[TIMING_CREATE_NEXT_GEN] / timing.total + 0.5));

    // free memory and return status  
    ga_test_complete();
    return ret;
}

void ga_learn_choose_best_answers(
    void)
{
    uint32_t   img_idx, ans;
    ans_nn_t   best_ans_nn;

    // since we are choosing a new set of best answers for all images,
    // clear the current answers
    for (ans = 0; ans < pop->max_answer; ans++) {
        pop->answer[ans].nn = -1;
    }

    // for all images being learned
    // - if the image is not a primary image then continue
    // - find the best_ans_nn 
    // - set pop->answer[ans].nn to the best_ans_nn found
    // endfor
    for (img_idx = 0; img_idx < pop->max_image_tbl; img_idx++) {
        if (pop->image_is_primary[img_idx] == FALSE) {
            continue;
        }

        (best_ans_nn = util_choose_best_ans_nn(img_idx));
        if (best_ans_nn == -1) {
            PRINT_ERROR("failed to find best_ans_nn\n");
            continue;
        }

        ans = pop->image_to_ans[img_idx];
        ASSERT_MSG(ans < pop->max_answer, "ans = %d", ans);
        pop->answer[ans].nn = best_ans_nn;
    }
}

void ga_learn_sort_orgs(
    void)
{
    uint32_t org_idx1, org_idx2, i;

    for (org_idx1 = 0; org_idx1 < pop->param_max_org; org_idx1++) {
        org_idx2 = score->org_weighted_score_sort[org_idx1].idx;

        if (org_idx2 == org_idx1) {
            continue;
        }

        util_swap(pop->org[org_idx1], pop->org[org_idx2], 
                  pop->max_chromes);
        util_swap(score->org_answer[org_idx1], score->org_answer[org_idx2],
                  sizeof(score_org_ans_t) * score->max_image_tbl);
        util_swap(&score->org_weighted_score[org_idx1], &score->org_weighted_score[org_idx2],
                  sizeof(score->org_weighted_score[0]));
        util_swap(&score->org_score[org_idx1], &score->org_score[org_idx2],
                  sizeof(score->org_score[0]));

        for (i = org_idx1; i < pop->param_max_org; i++) {
            if (score->org_weighted_score_sort[i].idx == org_idx1) {
                score->org_weighted_score_sort[i].idx = org_idx2;
            } else if (score->org_weighted_score_sort[i].idx == org_idx2) {
                score->org_weighted_score_sort[i].idx = org_idx1;
            }

            if (score->org_score_sort[i].idx == org_idx1) {
                score->org_score_sort[i].idx = org_idx2;
            } else if (score->org_score_sort[i].idx == org_idx2) {
                score->org_score_sort[i].idx = org_idx1;
            }
        }
    }

    // sanity check that org_weighted_score_sort is correct
    for (i = 0; i < pop->param_max_org; i++) {
        ASSERT_MSG(score->org_weighted_score_sort[i].idx == i, NULL);
    }
}

void ga_learn_create_next_gen(
    void)
{
    typedef struct {
        uint32_t p1_idx;
        uint32_t p2_idx;
        uint32_t sort_key;
    } parent_pair_t;

    double        sum_score[MAX_ORG], total_score;
    uint32_t      org_is_parent_cnt[MAX_ORG];
    parent_pair_t parent_pair_tbl[MAX_ORG/2];
    uint32_t      max_parent_pair_tbl;
    uint32_t      org_idx_avail[MAX_ORG];
    uint32_t      max_org_idx_avail;
    void        * temp_child[MAX_ORG];
    uint32_t      max_temp_child;
    uint32_t      org_idx, i;

    ASSERT_MSG(pop->param_max_org == score->max_org,
               "pop->param_max_org=%d score->max_org=%d",
               pop->param_max_org, score->max_org);

    // init
    bzero(sum_score, sizeof(sum_score));
    bzero(org_is_parent_cnt,sizeof(org_is_parent_cnt));
    bzero(parent_pair_tbl, sizeof(parent_pair_tbl));
    bzero(org_idx_avail, sizeof(org_idx_avail));
    bzero(temp_child, sizeof(temp_child));
    max_parent_pair_tbl = 0;
    max_org_idx_avail = 0;
    max_temp_child = 0;

    // sum the scores, this will be used by roulette selection algorithm below
    for (org_idx = 0; org_idx < score->max_org; org_idx++) {
        sum_score[org_idx] = 
            (org_idx == 0 ? 0. : sum_score[org_idx-1]) +
            1000. * score->org_weighted_score[org_idx];
    }
    total_score = 0;
    for (org_idx = 0; org_idx < score->max_org; org_idx++) {
        total_score += 1000. * score->org_weighted_score[org_idx];
    }
    ASSERT_MSG(total_score == sum_score[score->max_org-1],
               "total_score=%.1f sum_score[last]=%.1f", 
               total_score, sum_score[score->max_org-1]);

    // verify total_score is non zero
    if (total_score == 0) {
        PRINT_ERROR("total_score equals 0, can't create next gen\n");
        return;
    }

    // loop until all parent_pairs are selected for the next generation
    while (TRUE) {
        uint32_t rnd1, rnd2, org_idx, p1_idx, p2_idx, tries;

        // pick 2 parents using roulette wheel selection; 
        tries = 0;
        while (TRUE) {
            // select the 2 parent candidates
            rnd1 = util_random(total_score-1);
            rnd2 = util_random(total_score-1);
            p1_idx = p2_idx = -1;
            for (org_idx = 0; org_idx < pop->param_max_org; org_idx++) {
                if (rnd1 < sum_score[org_idx]) {
                    p1_idx = org_idx;
                    if (p2_idx != -1) {
                        break;
                    }
                }
                if (rnd2 < sum_score[org_idx]) {
                    p2_idx = org_idx;
                    if (p1_idx != -1) {
                        break;
                    }
                }
            }
            ASSERT_MSG(p1_idx != -1 && p2_idx != -1, NULL);

            // if the parents are different then we're done
            if (p1_idx != p2_idx) {
                break;
            }

            // if we've tried this too many times then something is wrong
            if (++tries > 100) {
                PRINT_ERROR("failed to select different parents, parent_idx=%d\n", p1_idx);
                break;
            }
        }

        // keep track of how many times each org_idx is being used as a parent
        org_is_parent_cnt[p1_idx]++;
        org_is_parent_cnt[p2_idx]++;
    
        // add parent pair to list, if list is full then break
        parent_pair_tbl[max_parent_pair_tbl].p1_idx = p1_idx;
        parent_pair_tbl[max_parent_pair_tbl].p2_idx = p2_idx;
        max_parent_pair_tbl++;
        if (max_parent_pair_tbl == pop->param_max_org/2) {
            break;
        }
    }

#if 0
    // log the percent of population used as parents 
    uint32_t org_used_total = 0;
    for (org_idx = 0; org_idx < pop->param_max_org; org_idx++) {
        if (org_is_parent_cnt[org_idx] != 0) {
            org_used_total++;
        }
    }
    util_pop_log("create_next_gen percent of pop used as parents = %0.1f",
                 100. * org_used_total / pop->param_max_org);
#endif

    // sort the parent_pair_tbl
    for (i = 0; i < max_parent_pair_tbl; i++) {
        parent_pair_tbl[i].sort_key = 
            org_is_parent_cnt[parent_pair_tbl[i].p1_idx] + 
            org_is_parent_cnt[parent_pair_tbl[i].p2_idx];
    }
    util_sort(parent_pair_tbl, max_parent_pair_tbl,
              sizeof(parent_pair_t), offsetof(parent_pair_t,sort_key), SORT_KEY_TYPE_UINT32);

    // init the org_idx_avail array
    for (org_idx = 0; org_idx < pop->param_max_org; org_idx++) {
        if (org_is_parent_cnt[org_idx] == 0) {
            org_idx_avail[max_org_idx_avail++] = org_idx;
        }
    }

    // loop over the parent_pair_tbl  
    for (i = max_parent_pair_tbl-1; i != -1; i--) {
        uint32_t p1_idx, p2_idx;
        void    *p1, *p2, *c1, *c2;

        // extract a parent pair
        p1_idx = parent_pair_tbl[i].p1_idx;
        p2_idx = parent_pair_tbl[i].p2_idx;
        p1     = pop->org[p1_idx];
        p2     = pop->org[p2_idx];

        // get 2 children from org_idx_avail
        // if org_idx_avail array is empty then malloc temp buffer
        if (max_org_idx_avail > 0) {
            c1 = pop->org[org_idx_avail[--max_org_idx_avail]];
        } else {
            c1 = temp_child[max_temp_child++] = malloc(pop->max_chromes);
        }
        if (max_org_idx_avail > 0) {
            c2 = pop->org[org_idx_avail[--max_org_idx_avail]];
        } else {
            c2 = temp_child[max_temp_child++] = malloc(pop->max_chromes);
        }

        // create the children
        if (c1 != NULL && c2 != NULL) {
            ga_learn_create_children(p1, p2, c1, c2);
        } else {
            PRINT_ERROR("failed alloc temp_child\n");
        }

        // decrement org_is_parent_cnt for each parent just used;  
        // if they won't be parent's again then add them to the org_idx_avail array
        if (org_is_parent_cnt[p1_idx] == 0 || org_is_parent_cnt[p2_idx] == 0) {
            PRINT_ERROR("org_is_parent_cnt is invalid, p1_idx=%d p2_idx=%d\n", p1_idx, p2_idx);
        }
        org_is_parent_cnt[p1_idx]--;
        if (org_is_parent_cnt[p1_idx] == 0) {
            org_idx_avail[max_org_idx_avail++] = p1_idx;
        }
        org_is_parent_cnt[p2_idx]--;
        if (org_is_parent_cnt[p2_idx] == 0) {
            org_idx_avail[max_org_idx_avail++] = p2_idx;
        }
    }

    // debug code to verify that org_is_parent_cnt is now zero 
    for (org_idx = 0; org_idx < pop->param_max_org; org_idx++) {
        if (org_is_parent_cnt[org_idx] != 0) {
            PRINT_ERROR("org_is_parent_cnt[%d] = %d\n",
                org_idx, org_is_parent_cnt[org_idx]);
        }
    }

    // should now be the same number of free slots as entries in the overflow table;
    // transfer the overflow table entries to the free slots
    if (max_org_idx_avail == max_temp_child) {
        for (i = 0; i < max_org_idx_avail; i++) {
            memcpy(pop->org[org_idx_avail[i]], temp_child[i], pop->max_chromes);
        }
    } else {
        PRINT_ERROR("max_org_idx_avail=%d max_temp_child=%d\n",
                    max_org_idx_avail, max_temp_child);
    }

    // free allocated temp_child
    for (i = 0; i < max_temp_child; i++) {
        free(temp_child[i]);
        temp_child[i] = NULL;
    }
}

void ga_learn_create_children(
    char * parent1, 
    char * parent2, 
    char * child1, 
    char * child2)
{
    bool     do_crossover;
    uint32_t mask, offset, bitloc;
    uint32_t max_chromes        = pop->max_chromes;

    // children are created by combining and mutating the parent chromes

    // start by making children exact copies of the parents
    memcpy(child1, parent1, max_chromes);
    memcpy(child2, parent2, max_chromes);

    // determine if crossover should be performed
    do_crossover = util_random(99) < SETTING_CROSSOVER_RATE_PERCENT;

    // perform crossover if flag set
    if (do_crossover) {
        // determine location of crossover, the crossover location is
        // constrained to be within the current top level chromosomes
        if (pop->neuron_lvl == 0) {
            bitloc = util_random(pop->max_chromes_lvl[0] * 8 - 1);
        } else if (pop->neuron_lvl == 1) {
            bitloc = pop->max_chromes_lvl[0] * 8 + 
                     util_random(pop->max_chromes_lvl[1] * 8 - 1);
        } else if (pop->neuron_lvl == 2) {
            bitloc = pop->max_chromes_lvl[0] * 8 + 
                     pop->max_chromes_lvl[1] * 8 + 
                     util_random(pop->max_chromes_lvl[2] * 8 - 1);
        } else {
            ASSERT_MSG(false, "neuron_lvl %d", pop->neuron_lvl);
        }

        // crossover first byte, this is usually a portion of a byte
        mask = (1 << (8 - (bitloc % 8))) - 1;
        offset = bitloc / 8;
        child1[offset] = (child1[offset] & ~mask) | (parent2[offset] & mask);
        child2[offset] = (child2[offset] & ~mask) | (parent1[offset] & mask);

        // crossover remaining full bytes
        offset = bitloc / 8 + 1;
        memcpy(child1+offset, parent2+offset, max_chromes-offset);
        memcpy(child2+offset, parent1+offset, max_chromes-offset);
    }
}

// -----------------  GA TEST-----------------------------------------

int32_t ga_test(
    image_t * image_tbl, 
    uint32_t max_image_tbl,
    uint32_t max_org)
{
    uint32_t i;

    // verify max_image_tbl is not zero
    if (max_image_tbl == 0) {
        PRINT_ERROR("no images supplied\n");
        return -1;
    }

    // verify pop->neuron_lvl is okay
    if (pop->neuron_lvl < 0 || pop->neuron_lvl >= MAX_NEURON_LVL) {
        PRINT_ERROR("invalid pop neuron_lvl %d\n", pop->neuron_lvl);
        return -1;
    }

    // allocate memory for ga_test_give/score
    if (ga_test_init(image_tbl, max_image_tbl, max_org) == -1) {
        PRINT_ERROR("ga_test_init failed\n");
        return -1;
    }

    // call ga_test_give to give the test
    if (ga_test_give() == -1) {
        PRINT_ERROR(" - interrupted\n");
        ga_test_complete();
        return -1;
    }

    // call ga_test_score to score the test
    ga_test_score();
    printf("\n");

    // print results ...

    // print scores, max,avg,min
    printf("SCORE            MAX   AVG   MIN\n");
    printf("-----            ---   ---   ---\n");
    printf("org            %5.1f %5.1f %5.1f\n",
           score->org_score_max, score->org_score_avg, score->org_score_min);
    printf("org_weighted   %5.1f %5.1f %5.1f\n",
           score->org_weighted_score_max, score->org_weighted_score_avg, score->org_weighted_score_min);
    printf("\n");

    // plot historgrams
    printf("SCORE HISTOGRAMS\n");
    printf("----------------\n");
    util_plot_histogram_org_score();
    util_plot_histogram_org_weighted_score();

    // print sorted orgs score and other info
    printf(" ORG_IDX     WSCORE     SCORE      CORRECT              DONT_KNOW            INCORRECT\n");
    printf(" -------     ------     -----      -------              ---------            ---------\n");
    for (i = 0; i < score->max_org; i++) {
        uint32_t org_idx;
        char s1[100], s2[100], s3[100];

        if (SETTING_VERBOSE ||
            i < 5 || 
            i > score->max_org - 5 ||
            (i % 50) == 0)
        {
            org_idx = score->org_weighted_score_sort[i].idx;
            printf("%8d %10.3f %10.3f     %-20s %-20s %-20s\n", 
                org_idx,
                score->org_weighted_score_sort[i].score,
                score->org_score[org_idx],
                ga_test_score_org_ans_info_str(org_idx,ANSWER_IS_CORRECT,s1),
                ga_test_score_org_ans_info_str(org_idx,ANSWER_IS_DONT_KNOW,s2),
                ga_test_score_org_ans_info_str(org_idx,ANSWER_IS_INCORRECT,s3));
        }
    }

    // free memory and return success
    ga_test_complete();
    return 0;
}

char * ga_test_score_org_ans_info_str(
    uint32_t org_idx,
    uint32_t ans_status,
    char * s)
{
    uint32_t count = 0;
    uint32_t sum_inc_neuron_count = 0;
    uint32_t min_inc_neuron_count = 1000000;
    uint32_t max_inc_neuron_count = 0;
    uint32_t img_idx;

    for (img_idx = 0; img_idx < score->max_image_tbl; img_idx++) {
        score_org_ans_t * oa = &score->org_answer[org_idx][img_idx];

        if (oa->ans_status != ans_status) {
            continue;
        }

        count++;
        sum_inc_neuron_count += oa->incorrect_neuron_count;
        if (oa->incorrect_neuron_count < min_inc_neuron_count) {
            min_inc_neuron_count = oa->incorrect_neuron_count;
        }
        if (oa->incorrect_neuron_count > max_inc_neuron_count) {
            max_inc_neuron_count = oa->incorrect_neuron_count;
        }
    }

    if (count == 0) {
        sprintf(s, "0");
    } else {
        sprintf(s, "%d(%d,%0.1f,%d)",
                count,
                min_inc_neuron_count,
                (double)sum_inc_neuron_count/count,
                max_inc_neuron_count);
    }

    return s;
}

int32_t  ga_test_init(
    image_t * image_tbl,
    uint32_t max_image_tbl,
    uint32_t max_org)
{
    size_t   score_size;
    uint32_t org_idx;

    // check param 
    if (max_image_tbl == 0) {
        PRINT_ERROR("max_image_tbl is zero\n");
        return -1;
    }

    // if score is already allocated then return error
    if (score != NULL) {
        PRINT_ERROR("score is already allocated\n");
        return -1;
    }

    // alloc score
    score_size = SCORE_SIZE(max_org, max_image_tbl);
    score = malloc(score_size);
    if (score == NULL) {
        PRINT_ERROR("alloc score, size %ld MB\n", score_size/MB);
        return -1;
    }
    bzero(score, score_size);
    printf("alloced score, size %ld MB\n", score_size/MB);

    // init non-zero fields of score
    score->image_tbl = image_tbl;
    score->max_image_tbl = max_image_tbl;
    score->max_org = max_org;
    for (org_idx = 0; org_idx < max_org; org_idx++) {
        score->org_answer[org_idx] =
                (void*)score + sizeof(score_t) +
                max_org * sizeof(void *) +
                org_idx * score->max_image_tbl * sizeof(score_org_ans_t);
    }

    // return success
    return 0;
}

void ga_test_complete(
    void)
{
    if (score == NULL) {
        return;
    }

    free(score);
    score = NULL;
}

int32_t ga_test_give(
    void)
{
    uint32_t percent_complete, percent_complete_last=-1;

    // asserts
    ASSERT_MSG(score != NULL, NULL);

    // verify pop->neuron_lvl
    if (pop->neuron_lvl < 0 || pop->neuron_lvl >= MAX_NEURON_LVL) {
        PRINT_ERROR("ga_test_give detected invalid pop->neuron_lvl %d\n", pop->neuron_lvl);
        return -1;
    }

    // call barrier to release the helper threads
    wt_req.req = WORK_THREAD_REQ_GIVE_TEST;
    wt_req.u.give_test.org_idx = 0;
    pthread_barrier_wait(&wt_barrier);

    // loop testing orgs
    printf("give_test     ");
    while (TRUE) {
        // if ctrl_c or no more then break
        if (ctrl_c || wt_req.u.give_test.org_idx >= score->max_org) {
            break;
        }

        // print progress
        usleep(100000);
        percent_complete = 100 * (wt_req.u.give_test.org_idx + 1) / score->max_org;
        if (percent_complete != percent_complete_last) {
            printf("\b\b\b\b%3d%%", percent_complete);
            percent_complete_last = percent_complete;
        }
    }

    // call barrier to ensure all threads are done prior to returning
    pthread_barrier_wait(&wt_barrier);

    // if interrupted then return error
    if (ctrl_c) {
        ctrl_c = FALSE;
        return -1;
    }

    // print final progress and return success
    percent_complete = 100;
    printf("\b\b\b\b%3d%%", percent_complete);
    return 0;
}

int32_t ga_test_score(
    void)
{
    uint32_t org_idx, img_idx;
    int32_t  points;
    uint32_t max_points;
    uint32_t percent_complete, percent_complete_last=-1;
    uint32_t best_ans;

    // asserts
    ASSERT_MSG(score != NULL, NULL);

    // print 'scoring'
    printf(" - scoring ");

    // clear find_best_answer lookup, and 
    // preset the lookup table entires that are for an exact match with an answer
    bzero(score->fba_lookup_tbl, sizeof(score->fba_lookup_tbl));
    for (best_ans = 0; best_ans < pop->max_answer; best_ans++) {
        ans_nn_t ans_nn;
        score_fba_lookup_ent_t * lte;

        ans_nn = pop->answer[best_ans].nn;
        lte = &score->fba_lookup_tbl[ans_nn % MAX_SCORE_FBA_LOOKUP_TBL];
        lte->ans_nn = ans_nn;
        lte->best_ans = best_ans;
        lte->best_ans_incorrect_neuron_cnt = 0;
        lte->state = 2;
    }

    // call barrier to release the helper threads
    wt_req.req = WORK_THREAD_REQ_SCORE_TEST;
    wt_req.u.score_test.org_idx = 0;
    pthread_barrier_wait(&wt_barrier);

    // loop, scoring orgs
    printf("    ");
    while (TRUE) {
        // if ctrl_c or no more then break
        if (ctrl_c || wt_req.u.give_test.org_idx >= score->max_org) {
            break;
        }

        // print progress
        usleep(100000);
        percent_complete = 100 * (wt_req.u.give_test.org_idx + 1) / score->max_org;
        if (percent_complete != percent_complete_last) {
            printf("\b\b\b\b%3d%%", percent_complete);
            percent_complete_last = percent_complete;
        }
    }

    // call barrier to ensure all threads are done prior to proceeding
    pthread_barrier_wait(&wt_barrier);

    // if interrupted then return error
    if (ctrl_c) {
        ctrl_c = FALSE;
        return -1;
    }

    // print final percent complete
    percent_complete = 100;
    printf("\b\b\b\b%3d%%", percent_complete);

    // determine org_score
    for (org_idx = 0; org_idx < score->max_org; org_idx++) {
        points = max_points = 0;
        for (img_idx = 0; img_idx < score->max_image_tbl; img_idx++) {
            if (score->org_answer[org_idx][img_idx].ans_status == ANSWER_IS_CORRECT) {
                points++;
            }
            max_points++;
        }
        score->org_score[org_idx] = (double)points / max_points * 100;
    }

    // determine org_weighted_score
    for (org_idx = 0; org_idx < score->max_org; org_idx++) {
        points = max_points = 0;
        for (img_idx = 0; img_idx < score->max_image_tbl; img_idx++) {
            if (score->org_answer[org_idx][img_idx].ans_status == ANSWER_IS_CORRECT) {
                int32_t tmp;
                if (MAX_INCORRECT_NEURON_CNT > 0) {
                    tmp = 100 - (score->org_answer[org_idx][img_idx].incorrect_neuron_count * 100 / 
                                 MAX_INCORRECT_NEURON_CNT);
                } else {
                    tmp = 100;
                }
                ASSERT_MSG(tmp >= 0 && tmp <= 100, "tmp=%d", tmp);
                points += tmp;
            } 
            max_points += 100;
        }
        score->org_weighted_score[org_idx] = (double)points / max_points * 100;
        if (score->org_weighted_score[org_idx] < 1) {
            score->org_weighted_score[org_idx] = 1;
        }
    }

    // determine score stats
    ga_test_score_compute_stats(score->org_score, score->max_org,
                                &score->org_score_max, 
                                &score->org_score_avg, 
                                &score->org_score_min);
    ga_test_score_compute_stats(score->org_weighted_score, score->max_org,
                                &score->org_weighted_score_max, 
                                &score->org_weighted_score_avg, 
                                &score->org_weighted_score_min);

    // determine sorted org scores
    for (org_idx = 0; org_idx < score->max_org; org_idx++) {
        score->org_score_sort[org_idx].sort_key = 
            1000000000. * score->org_score[org_idx] + 
            (MAX_ORG - org_idx);
        score->org_score_sort[org_idx].score = score->org_score[org_idx];
        score->org_score_sort[org_idx].idx = org_idx;
    }
    util_sort(score->org_score_sort, score->max_org,
              sizeof(score_sort_t), offsetof(score_sort_t,sort_key), SORT_KEY_TYPE_UINT64);

    // determine sorted org weighted_scores
    for (org_idx = 0; org_idx < score->max_org; org_idx++) {
        score->org_weighted_score_sort[org_idx].sort_key = 
            1000000000. * score->org_weighted_score[org_idx] + 
            (MAX_ORG - org_idx);
        score->org_weighted_score_sort[org_idx].score = score->org_weighted_score[org_idx];
        score->org_weighted_score_sort[org_idx].idx = org_idx;
    }
    util_sort(score->org_weighted_score_sort, score->max_org,
              sizeof(score_sort_t), offsetof(score_sort_t,sort_key), SORT_KEY_TYPE_UINT64);

    // return success
    return 0;
}

void ga_test_score_find_best_answer(
    ans_nn_t ans_nn,
    uint32_t * best_ans_arg,
    uint32_t * best_ans_incorrect_neuron_cnt_arg)
{
    uint32_t best_ans;
    uint32_t best_ans_incorrect_neuron_cnt;
    uint32_t i, bits_diff;
    score_fba_lookup_ent_t * lte;

    // asserts
    ASSERT_MSG(sizeof(ans_nn_t) == 16, NULL);

    // check if ans_nn is in the find-best-answer-lookup table, if so return 
    // best_ans and best_ans_incorrect_neuron_cnt from lookup table
    lte = &score->fba_lookup_tbl[ans_nn % MAX_SCORE_FBA_LOOKUP_TBL];
    if (lte->state == 2 && lte->ans_nn == ans_nn) {
        *best_ans_arg = lte->best_ans;
        *best_ans_incorrect_neuron_cnt_arg = lte->best_ans_incorrect_neuron_cnt;
        return;
    }

    // if MAX_INCORRECT_NEURON_CNT is zero and the above lookup failed then
    // there are 2 possibilities:
    // - if there is no entry in the lookup table then the ans_nn is wrong
    // - otherwise there was a collision when the lookup table was initialized,
    //   so we need to search the answer table for a matching ans_nn (this should
    //   not happen often)
    if (MAX_INCORRECT_NEURON_CNT == 0) {
        if (lte->state == 0) {
            *best_ans_arg = -1;
            *best_ans_incorrect_neuron_cnt_arg = 255; 
        } else {
            // printf("MUST HAVE BEEN A COLLISION IN LOOKUP TABLE \n");
            *best_ans_arg = -1;
            *best_ans_incorrect_neuron_cnt_arg = 255; 
            for (i = 0; i < pop->max_answer; i++) {
                if (pop->answer[i].nn == ans_nn) {
                    *best_ans_arg = i;  
                    *best_ans_incorrect_neuron_cnt_arg = 0; 
                    break;
                }
            }
        }
        return;
    }

    // search all answers for best_ans
    best_ans = 0;
    best_ans_incorrect_neuron_cnt = sizeof(ans_nn_t) * 8 + 1;
    for (i = 0; i < pop->max_answer; i++) {
        bits_diff = num_bits_diff(ans_nn, pop->answer[i].nn);
        if (bits_diff < best_ans_incorrect_neuron_cnt) {
            best_ans_incorrect_neuron_cnt = bits_diff;
            best_ans = i;
        }
    }

    // add to hash table 
    if (__sync_bool_compare_and_swap(&lte->state, 0, 1)) {
        lte->ans_nn = ans_nn;
        lte->best_ans = best_ans;
        lte->best_ans_incorrect_neuron_cnt = best_ans_incorrect_neuron_cnt;
        __sync_synchronize();
        lte->state = 2;
    }

    // return best_ans and best_ans_incorrect_neuron_cnt
    *best_ans_arg = best_ans;
    *best_ans_incorrect_neuron_cnt_arg = best_ans_incorrect_neuron_cnt;
}

void ga_test_score_compute_stats(
    double * values, 
    uint32_t max_values,
    double * max_arg, 
    double * avg_arg, 
    double * min_arg)
{
    double sum=0, min=1000000, max=-1000000, count=0;
    uint32_t i;

    for (i = 0; i < max_values; i++) {
        if (values[i] == -1) {
            continue;
        }
        sum += values[i];
        if (values[i] < min) {
            min = values[i];
        }
        if (values[i] > max) {
            max = values[i];
        }
        count++;
    }

    if (count > 0) {
        *max_arg = max;
        *avg_arg = sum / count;
        *min_arg = min;
    } else {
        *max_arg = 0;
        *avg_arg = 0;
        *min_arg = 0;
    }
}

// -----------------  GENETIC ALGORITHM DISPLAY  ---------------------

void ga_display(
    void)
{
    uint32_t i, j;

    printf("CURRENT VALUES\n"
           "--------------\n"
           "    gen_num                        = %d\n"
           "    org_score                      = %0.1f %0.1f %0.1f\n"
           "    org_weighted_score             = %0.1f %0.1f %0.1f\n"
           "    max_image_tbl                  = %d\n"
           "    max_org                        = %d\n"
           "    neuron_lvl                     = %d\n"
           "    max_chromes                    = %d (%d + %d + %d)\n"
           "    max_answer                     = %d\n"
           "\n"
           "    param_max_org                  = %d\n"
           "    param_max_neuron               = %d %d %d\n"
           "    param_max_incorrect_neuron_cnt = %d %d %d\n"
           "    param_image_filename           = %s",
           pop->gen_num,
           pop->org_score_max, pop->org_score_avg, pop->org_score_min,
           pop->org_weighted_score_max, pop->org_weighted_score_avg, pop->org_weighted_score_min,
           pop->max_image_tbl,
           pop->param_max_org,
           pop->neuron_lvl,
           pop->max_chromes, pop->max_chromes_lvl[0], pop->max_chromes_lvl[1], pop->max_chromes_lvl[2],
           pop->max_answer,
           pop->param_max_org,
           pop->param_max_neuron[0],
           pop->param_max_neuron[1],
           pop->param_max_neuron[2],
           pop->param_max_incorrect_neuron_cnt[0],
           pop->param_max_incorrect_neuron_cnt[1],
           pop->param_max_incorrect_neuron_cnt[2],
           pop->param_image_filename);
    printf("\n");

    // print log
    printf("LOG\n"
           "---\n"
           "%s",
           pop->log);
    printf("\n");

    // print histograms
    // - ans_nn distribution histogram
    printf("POP HISTOGRAMS\n");
    printf("--------------\n");
    util_plot_histogram_ans_nn();

    // print answer and image info
    printf("  ANSWER  ANSWER_NAME                           ANS_NN  IMAGE_FILE_NAMES ...\n");
    printf("  ------  -----------                           ------ --------------------\n");
    for (i = 0; i < pop->max_answer; i++) {

        if (SETTING_VERBOSE == 0 && i >= 10) {
            printf ("      ..................\n");
            break;
        }

        printf("%8d %12s %32s ",
               i,
               pop->answer[i].name,
               util_ans_nn_to_str(pop->answer[i].nn,16));
        for (j = 0; j < pop->max_image_tbl; j++) {
            if (strcmp(pop->image_tbl[j].ans_name, pop->answer[i].name) == 0) {
                printf("%s ", pop->image_tbl[j].filename);
            }
        }
        printf("\n");
    }
    printf("\n");
}

// -----------------  NEURAL NET  ------------------------------------

ans_nn_t nn_eval(
    uint32_t org_idx, 
    char * pixels)
{
    #define NN_ZERO 0
    #define NN_ONE  1

    uint32_t * max_neuron;
    uint32_t   max_input, max_output=0, lvl, n, i;
    char     * inputs;
    char       outputs[MAX_ANSWER_NN_BITS];
    char     * chromes;
    ans_nn_t   ans_nn;
    int32_t    neuron_lvl;
    uint32_t   max_chromes;

    // check neuron_lvl
    ASSERT_MSG(pop->neuron_lvl >= 0 && pop->neuron_lvl <= MAX_NEURON_LVL, 
               "pop->neuron_lvl %d", pop->neuron_lvl);

    max_neuron     = pop->param_max_neuron;
    chromes        = pop->org[org_idx];
    neuron_lvl     = pop->neuron_lvl;
    max_input      = MAX_PIXELS;
    inputs         = pixels;

    max_chromes = 0;
    for (i = 0; i <= neuron_lvl; i++) {
        max_chromes += pop->max_chromes_lvl[i];
    }

    // loop over neural net levels
    for (lvl = 0; lvl <= neuron_lvl; lvl++) {
        int32_t   sum;
        char    * weights;

        // determine output for all neurons at this lvl
        for (n = 0; n < max_neuron[lvl]; n++) {
            if (lvl == 0) {
                // compute sum of weights * input_value
                weights = (char*)chromes;
                sum = 0;
                for (i = 0; i < max_input; i++) {
                    sum += weights[i*MAX_COLOR + inputs[i]];
                }

                // compute output 
                outputs[n] = (sum > 0 ? NN_ONE : NN_ZERO);

                // point chromes to the next neuron
                chromes += max_input * MAX_COLOR;
            } else {
                // compute sum of weights * input_value
                weights = (char*)chromes;
                sum = 0;
                for (i = 0; i < max_input; i++) {
                    if (inputs[i]) {
                        sum += weights[i];
                    }
                }

                // compute output 
                outputs[n] = (sum > 0 ? NN_ONE : NN_ZERO);

                // point chromes to the next neuron
                chromes += max_input;
            }
        }
        max_output = max_neuron[lvl];

        // set input for next level
        inputs = outputs;
        max_input = max_output;
    }

    // asserts:
    // - that all chromes have been used
    // - that there are not to many output neuron values to fit in the ans_nn_t type
    ASSERT_MSG(chromes - pop->org[org_idx] == max_chromes,
               "actual=%zd expected=%d",
               chromes - pop->org[org_idx],
               max_chromes);
    ASSERT_MSG(max_output <= MAX_ANSWER_NN_BITS, 
               "max_output=%d", 
               max_output);

    // construct return ans_nn value from last level processed
    ans_nn = 0;
    for (n = 0; n < max_output; n++) {
        if (outputs[n] == NN_ONE) {
            ans_nn |= ((ans_nn_t)1 << n);
        }
    }
    ASSERT_MSG(ans_nn <= MAX_ANS_NN(max_neuron[neuron_lvl]), 
               "ans_nn=%s max_ans_nn=%s", 
               util_ans_nn_to_str(ans_nn,16), 
               util_ans_nn_to_str(MAX_ANS_NN(max_neuron[neuron_lvl]),16));

    // return 
    return ans_nn;
}        

// -----------------  UTIL ROUTINES  ---------------------------------

// - - - - - - - - -  UTIL READ/WRITE POP FILE - - - - - - - - - - - - 

int32_t util_write_pop_file(
    char * pop_fn)
{
    int32_t fd;
    char    pop_fn_bak[MAX_FILE_NAME];
    size_t  pop_size = pop->pop_size;

    // verify pop_fn is supplied
    if (pop_fn[0] == '\0' ) {
        PRINT_ERROR("pop filename arg is required\n");
        return -1;
    }

    // rename the old pop_fn, if it exists, to pop_fn.bak
    sprintf(pop_fn_bak, "%s.bak", pop_fn);
    rename(pop_fn, pop_fn_bak);

    // open
    if ((fd = open(pop_fn, O_CREAT|O_EXCL|O_WRONLY, 0666)) < 0) {
        PRINT_ERROR("failed open %s, %s\n", pop_fn, strerror(errno));
        return -1;
    }

    // write
    if (write(fd, pop, pop_size) != pop_size) {
        PRINT_ERROR("failed write %s, %s\n", pop_fn, strerror(errno));
        close(fd);
        return -1;
    }

    // set pop_filename
    if (pop_filename != pop_fn) {
        strcpy(pop_filename, pop_fn);
    }

    // close
    close(fd);

    // print and return success
    printf("write %s succeeded\n", pop_fn);
    return 0;
}

int32_t util_read_pop_file(
    char * pop_fn)
{
    struct stat buf;
    int32_t     fd;
    size_t      pop_size;
    uint32_t    org_idx;

    // free the current pop, and clear pop_filename
    free(pop);
    pop = NULL;
    pop_filename[0] = '\0';

    // stat the pop_fn to get it's size
    if (stat(pop_fn, &buf) < 0) {
        PRINT_ERROR("failed stat %s\n", pop_fn);
        return -1;
    }
    pop_size = buf.st_size;

    // check pop_size is at least minimum
    if (pop_size < sizeof(pop_t)) {
        PRINT_ERROR("pop_size %zd is less than sizeof(pop_t) %zd\n",
                    pop_size, sizeof(pop_t));
        return -1;
    }

    // allocate pop
    pop = malloc(pop_size);
    if (pop == NULL) {
        PRINT_ERROR("alloc pop");
        return -1;
    }
    bzero(pop, pop_size);

    // read the pop_fn file
    if ((fd = open(pop_fn, O_RDONLY)) < 0) {
        PRINT_ERROR("failed open %s, %s\n", pop_fn, strerror(errno));
        free(pop);
        pop = NULL;
        return -1;
    }
    if (read(fd, pop, pop_size) != pop_size) {
        PRINT_ERROR("failed read %s, %s\n", pop_fn, strerror(errno));
        close(fd);
        free(pop);
        pop = NULL;
        return -1;
    }
    close(fd);

    // verify magic and size
    if (pop->magic != POP_MAGIC) {
        PRINT_ERROR("invalid magic in %s, magic=0x%x expected=0x%x\n",
                    pop_fn, pop->magic, POP_MAGIC);
        free(pop);
        pop = NULL;
        return -1;
    }
    if (pop_size != pop->pop_size) {
        PRINT_ERROR("size of file %s is %zd, expected %zd\n",
                    pop_fn, pop_size, pop->pop_size);
        free(pop);
        pop = NULL;
        return -1;
    }

    // fixup the pop->org pointers 
    for (org_idx = 0; org_idx < pop->param_max_org; org_idx++) {
        pop->org[org_idx] = (void*)pop + sizeof(pop_t) + 
                            pop->param_max_org* sizeof(void *) +
                            org_idx * pop->max_chromes;
    }

    // set pop_filename
    strcpy(pop_filename, pop_fn);

    // return success
    return 0;
}

// - - - - - - - - -  UTIL READ IMAGE FILES  - - - - - - - - - - - - - 

int32_t util_read_images_from_filename(
    char * image_filename,
    image_t ** image_tbl_ret, 
    uint32_t * max_image_tbl_ret)
{
    FILE           * fp = NULL;
    char             cmd_line[10000];
    uint32_t         max_image_tbl = 0;

    static uint32_t  max_image_tbl_alloced;
    static image_t * image_tbl;

    // preset return
    *max_image_tbl_ret = 0;

    // assert that pop is allocated and score is not allocated
    ASSERT_MSG(pop != NULL, NULL);
    ASSERT_MSG(score == NULL, NULL);

    // expand image_filename 
    sprintf(cmd_line, "echo -n %s", image_filename);
    if (madvise(PAGE_ALIGN(pop), pop->pop_size, MADV_DONTFORK) == -1) {
        PRINT_ERROR("failed madvise dontfork %s\n", strerror(errno));
    }
    fp = popen(cmd_line, "r");
    if (madvise(PAGE_ALIGN(pop), pop->pop_size, MADV_DOFORK) == -1) {
        PRINT_ERROR("failed madvise dofork %s\n", strerror(errno));
    }
    if (fp == NULL) {
        PRINT_ERROR("popen %s\n", strerror(errno));
        return -1;
    }

    // read in all files on the list if there is an error reading any 
    // of them then return error;  for each file read, check if the
    // image already exists in pop, if so then print msg and do not
    // include this image otherwise add this to image array
    while (TRUE) {
        image_t image;
        char image_fn[MAX_FILE_NAME];

        // get the next image filename; if no more then break
        image_fn[0] = '\0';
        fscanf(fp, "%s", image_fn);
        if (image_fn[0] == '\0') {
            break;
        }

        // read the image_fn into image
        if (util_read_image_file(image_fn, &image) < 0) {
            PRINT_ERROR("reading '%s'\n", image_fn);
            pclose(fp);
            return -1;
        }

        // realloc image_tbl if needed, increasing size by 100 entries
        if (max_image_tbl >= max_image_tbl_alloced) {
            max_image_tbl_alloced += 100;
            image_tbl = realloc(image_tbl, max_image_tbl_alloced*sizeof(image_t));
        }

        // add the image to the image_tbl that will be returned
        image_tbl[max_image_tbl++] = image;
    }
    printf("completed read of %d image files\n", max_image_tbl);

    // pclose
    pclose(fp);

    // return image_tbl and max_image_tbl
    *image_tbl_ret = image_tbl;
    *max_image_tbl_ret = max_image_tbl;
    return 0;
}

int32_t util_read_image_file(
    char * image_fn, 
    image_t * image)
{
    FILE   * fp;
    char     s[1000];
    uint32_t i;
    uint32_t pix_idx = 0, y = 0;
    char   * basename_image_fn;

    // clear image struct
    bzero(image, sizeof(image_t));

    // open the image_fn 
    fp = fopen(image_fn, "r");
    if (fp == NULL) {
        PRINT_ERROR("file '%s' open, %s\n", image_fn, strerror(errno));
        return -1;
    }

    // read lines from image_fn
    while (fgets(s, sizeof(s), fp) != NULL) {
        pix_idx = y * MAX_X;
        for (i = 0; i < MAX_X && s[i] != '\n'; i++) {
            switch (s[i]) {
            case '.': image->pixels[pix_idx++] = BLACK;   break;
            case 'R': image->pixels[pix_idx++] = RED;     break;
            case 'G': image->pixels[pix_idx++] = GREEN;   break;
            case 'Y': image->pixels[pix_idx++] = YELLOW;  break;
            case 'B': image->pixels[pix_idx++] = BLUE;    break;
            case 'M': image->pixels[pix_idx++] = MAGENTA; break;
            case 'C': image->pixels[pix_idx++] = CYAN;    break;
            case 'W': image->pixels[pix_idx++] = WHITE;   break;
            default:
                PRINT_ERROR("file '%s' invalid char at line %d col %d\n",
                            image_fn, y+1, i+1);
                fclose(fp);
                return -1;
            }
        }

        if (++y == MAX_Y) {
            break;
        }
    }
    
    // close image_fn
    fclose(fp);

    // set image->filename
    basename_image_fn = basename(image_fn);
    strncpy(image->filename, basename_image_fn, MAX_FILE_NAME);
    image->filename[MAX_FILE_NAME-1] = '\0';
    
    // set image->ans_name,
    // convert image_fn to ans_name by replacing the first '-' or '.' with '\0'; 
    strtok(basename_image_fn, "-.");
    if (strlen(basename_image_fn) > MAX_ANSWER_NAME-1) {
        PRINT_ERROR("ans_name '%s' is too long, max=%d\n",
                    basename_image_fn, MAX_ANSWER_NAME-1);
        return -1;
    }
    strcpy(image->ans_name, basename_image_fn);

    // return success
    return 0;
}

void util_print_pixel(
    uint32_t color)
{
    printf("%c[%dm  %c[39;49m", 0x1b, 100+color, 0x1b);
}

// - - - - - - - - -  UTIL RANDOM NUMBER SUPPORT - - - - - - - - - - - 

uint32_t util_random(
    uint32_t max_val)
{
    // returns random number in range 0 .. max_val  (inclusive)
    return ((uint64_t)max_val + 1) * (uint32_t)random() / ((uint64_t)RAND_MAX + 1);
}

void util_random_set_default_seed(
    void)
{
    srandom(1);
}

// - - - - - - - - -  UTIL HISTOGRAMS   - - - - - - - - - - - - - - - 

void util_plot_histogram(
    char * hist_name_str,
    double min_x,
    double max_x,
    uint32_t max_data,
    double * data)
{
    #define MAX_BUCKET 100
    #define MAX_LINE   5

    uint32_t bucket[MAX_BUCKET];
    uint32_t max_bkt_val;
    int32_t  i, j, bkt_idx;

    // init
    bzero(bucket, sizeof(bucket));
    max_bkt_val = 0;

    // put data in buckets
    for (i = 0; i < max_data; i++) {
        bkt_idx = (data[i] - min_x) / (max_x - min_x) * MAX_BUCKET;
        if (bkt_idx < 0) {
            bkt_idx = 0;
        } else if (bkt_idx >= MAX_BUCKET) {
            bkt_idx = MAX_BUCKET-1;
        }
        bucket[bkt_idx]++;
    }

    // determine max bucket value
    for (i = 0; i < MAX_BUCKET; i++) {
        if (bucket[i] > max_bkt_val) {
            max_bkt_val = bucket[i];
        }
    }

    // plot graph in char array
    for (i = MAX_LINE-1; i >= 0; i--) {
        if (i == MAX_LINE-1) {
            printf("%5d |", max_bkt_val);
        } else if (i == 0) {
            printf("%5d |", 0);
        } else {
            printf("      |");
        }
        for (j = 0; j < MAX_BUCKET; j++) {
            putchar(bucket[j] >= i * max_bkt_val / MAX_LINE + 1 ? 'X' : ' ');
        }
        putchar('\n');
    }
    printf("      +----------------------------------------------------------------------------------------------------\n");
    printf("      %-10.5g %30s %-48s %10.5g\n", min_x, "", hist_name_str, max_x);
    printf("\n");
}

void util_plot_histogram_org_score(
    void)
{
    util_plot_histogram("ORG_SCORE", 0, 100, score->max_org, score->org_score);
}

void util_plot_histogram_org_weighted_score(
    void)
{
    util_plot_histogram("ORG_WEIGHTED_SCORE", 0, 100, score->max_org, score->org_weighted_score);
}

void util_plot_histogram_ans_nn(
    void)
{
    double answer_nn[MAX_ANSWER];
    uint32_t i;

    for (i = 0; i < pop->max_answer; i++) {
        answer_nn[i] = pop->answer[i].nn;
    }
    
    util_plot_histogram("ANSWER_NN", 0, MAX_ANS_NN(pop->param_max_neuron[pop->neuron_lvl]), pop->max_answer, answer_nn);
}

// - - - - - - - - -  UTIL MERGE SORT - - - - - - - - - - - - - - - - 

void util_sort(
    void * array,
    uint32_t elements,
    uint32_t element_size,
    off_t sort_key_offset,
    uint32_t sort_key_type
    )
{
    int left_idx, right_idx, merge_idx;
    void *merged, *lv, *rv;
    bool lv_gt_rv;

    // if no sorting needed then return
    if (elements <= 1) {
        return;
    }

    // sort the left half and the right half seperately
    util_sort(array, elements/2, element_size, sort_key_offset, sort_key_type);
    util_sort(array+((elements/2)*element_size), elements-elements/2, element_size, sort_key_offset, sort_key_type);

    // merge the sorted left and right halfs into a single sorted list
    merged = malloc(elements*element_size);
    if (merged == NULL) {
        PRINT_ERROR("alloc merged\n");
        return;
    }
    left_idx = 0; 
    right_idx = elements/2;
    merge_idx = 0;
    while (TRUE) {
        // if merged array is full then we're done
        if (merge_idx == elements) {
            break;
        }

        // if nothing remains on the left side then 
        // copy the remaining from the right side, and
        // we're done
        if (left_idx == elements/2) {
            memcpy(merged+merge_idx*element_size, 
                   array+right_idx*element_size, 
                   (elements-merge_idx)*element_size);
            break;
        }

        // if nothing remains on the right side then 
        // copy the remaining from the left side, and
        // we're done
        if (right_idx == elements) {
            memcpy(merged+merge_idx*element_size, 
                   array+left_idx*element_size, 
                   (elements-merge_idx)*element_size);
            break;
        }

        // take value from the left side or right side as appropriate
        lv = array+left_idx*element_size;
        rv = array+right_idx*element_size;
        switch (sort_key_type) {
        case SORT_KEY_TYPE_DOUBLE:
            lv_gt_rv = (*(double*)(lv+sort_key_offset) > *(double*)(rv+sort_key_offset));
            break;
        case SORT_KEY_TYPE_UINT32:
            lv_gt_rv = (*(uint32_t*)(lv+sort_key_offset) > *(uint32_t*)(rv+sort_key_offset));
            break;
        case SORT_KEY_TYPE_UINT64:
            lv_gt_rv = (*(uint64_t*)(lv+sort_key_offset) > *(uint64_t*)(rv+sort_key_offset));
            break;
        case SORT_KEY_TYPE_UINT128:
            lv_gt_rv = (*(__uint128_t*)(lv+sort_key_offset) > *(__uint128_t*)(rv+sort_key_offset));
            break;
        default:
            lv_gt_rv = FALSE;
            ASSERT_MSG(FALSE, "sort_key_type=%d", sort_key_type);
            break;
        }
        if (lv_gt_rv) {
            memcpy(merged+merge_idx*element_size, 
                   array+left_idx*element_size, 
                   element_size);
            merge_idx++;
            left_idx++;
        } else {
            memcpy(merged+merge_idx*element_size, 
                   array+right_idx*element_size, 
                   element_size);
            merge_idx++;
            right_idx++;
        }
    }
    memcpy(array, merged, elements*element_size);
    free(merged);
}

// - - - - - - - - -  UTIL CHOOSE BEST_ANS_NN  - - - - - - - - - 

// #define DEBUG_UTIL_CHOOSE_BEST_ANS_NN

ans_nn_t util_choose_best_ans_nn(
    uint32_t img_idx)
{
    uint32_t i, j;

    ans_nn_t * ans_nn_sort;

    struct ans_tbl_s {
        ans_nn_t nn;
        uint32_t occurances;
        uint32_t min_bits_diff;
    } *ans_tbl;
    uint32_t max_ans_tbl;
    ans_nn_t tmp_ans_nn;

    uint32_t best_min_bits_diff;
    uint32_t best_occurances;
    ans_nn_t best_ans_nn;
    uint32_t best_ans_tbl_idx;

    // create sorted list of answers
    ans_nn_sort = calloc(pop->param_max_org, sizeof(ans_nn_t));
    if (ans_nn_sort == NULL) {
        PRINT_ERROR("calloc ans_nn_sort failed\n");
        return -1;
    }
    for (i = 0; i < pop->param_max_org; i++) {
        ans_nn_sort[i] = score->org_answer[i][img_idx].ans_nn;
    }
    util_sort(ans_nn_sort, 
              pop->param_max_org,
              sizeof(ans_nn_t), 
              0,
              SORT_KEY_TYPE_UINT128);
    
#ifdef DEBUG_UTIL_CHOOSE_BEST_ANS_NN
    printf("--------- START img_idx=%d ---------\n", img_idx);
    for (i = 0; i < pop->param_max_org; i++) {
        printf("ans_nn_sort: %s\n", 
              util_ans_nn_to_str(ans_nn_sort[i],16));
    }
    printf("\n");
#endif

    // construct a table containing:
    // - ans_nn
    // - occurances:  number of occurances of this answer
    // - min_bits_diff:  minimum number of bits different between this
    //   ans_nn and all other images ans_nn
    //
    // note that when this routine is called the pop->answer[].nn has
    // been reset; so need to skip these reset values when determining
    // min_bits_diff

    // - allocate ans_tbl
    ans_tbl = calloc(pop->param_max_org, sizeof(ans_tbl[0]));
    if (ans_tbl == NULL) {
        PRINT_ERROR("calloc ans_tbl failed\n");
        free(ans_nn_sort);
        return -1;
    }
    max_ans_tbl = 0;

    // - init ans_nn_tbl fields: ans_nn and occurances
    tmp_ans_nn = -1;
    for (i = 0; i < pop->param_max_org; i++) {
        if (ans_nn_sort[i] != tmp_ans_nn) {
            max_ans_tbl++;
            ans_tbl[max_ans_tbl-1].nn = ans_nn_sort[i];
            tmp_ans_nn = ans_nn_sort[i];
        }
        ans_tbl[max_ans_tbl-1].occurances++;
    }

    // - init ans_nn_tbl field min_bits_diff
    for (i = 0; i < max_ans_tbl; i++) {
        ans_tbl[i].min_bits_diff = 128;

        for (j = 0; j < pop->max_answer; j++) {
            if (pop->answer[j].nn == -1) {
                continue;
            }

            uint32_t bits_diff = num_bits_diff(ans_tbl[i].nn, pop->answer[j].nn);
            if (bits_diff < ans_tbl[i].min_bits_diff) {
                ans_tbl[i].min_bits_diff = bits_diff;
            }
        }
    }

#ifdef DEBUG_UTIL_CHOOSE_BEST_ANS_NN
    printf("max_ans_tbl = %d\n", max_ans_tbl);
    for (i = 0; i < max_ans_tbl; i++) {
        printf("%3d: occurances, min_bits_diff, nn = %2d %2d %s\n",
               i,
               ans_tbl[i].occurances,
               ans_tbl[i].min_bits_diff, 
               util_ans_nn_to_str(ans_tbl[i].nn,16));
    }
    printf("\n");
#endif

try_again:
    // init variables used to find the best answer
    best_occurances = 0;
    best_min_bits_diff = 0;
    best_ans_nn = -1;
    best_ans_tbl_idx = -1;

    // find highest occurances
    for (i = 0; i < max_ans_tbl; i++) {
        if (ans_tbl[i].occurances > best_occurances) {
            best_occurances = ans_tbl[i].occurances;
        }
    }

    // if best_occurances is zero, this means all answers were
    // found to be duplicates (via the goto try_again below)
    if (best_occurances == 0) {
        PRINT_ERROR("ALL DUPLICATES\n");
        best_ans_nn = -1;
        goto done;
    }

    // find best min_bits_diff for the ans_tbl entries with the best_occurances
    for (i = 0; i < max_ans_tbl; i++) {
        if (ans_tbl[i].occurances != best_occurances) {
            continue;
        }
        if (ans_tbl[i].min_bits_diff >= best_min_bits_diff) {
            best_min_bits_diff = ans_tbl[i].min_bits_diff;
            best_ans_nn = ans_tbl[i].nn;
            best_ans_tbl_idx   = i;
        }
    }

    // if the best answer found is a duplicate of an existing answer then
    // clear ans_tbl[].occurances so this answer won't be found again, and
    // try again
    if (best_min_bits_diff == 0) {
#ifdef DEBUG_UTIL_CHOOSE_BEST_ANS_NN
        printf("best_min_bit_diff is zero, trying again\n");
#endif
        ans_tbl[best_ans_tbl_idx].occurances  = 0;
        goto try_again;
    }

    // check for duplicate
    if (best_min_bits_diff == 0) {
        PRINT_ERROR("best_min_bits_diff is zero: img_idx=%d occurances=%d max_ans_tbl=%d best_ans_nn=%s\n",
                    img_idx, best_occurances, max_ans_tbl, util_ans_nn_to_str(best_ans_nn,16));
    }

    // debug prints, and enter <cr> to continue
#ifdef DEBUG_UTIL_CHOOSE_BEST_ANS_NN
    printf("best_ans_tbl_idx=%d best_min_bits_diff=%d best_occurances=%d best_ans_nn=%s\n",
        best_ans_tbl_idx, best_min_bits_diff, best_occurances,
        util_ans_nn_to_str(best_ans_nn,16));
    printf("--------- DONE img_idx=%d ---------\n", img_idx);
    printf("enter to continue ...");
    char s[100];
    fgets(s,sizeof(s),stdin);
#endif

done:
    // free
    free(ans_nn_sort);
    free(ans_tbl);

    // return best_ans_nn
    return best_ans_nn;
}

// - - - - - - - - -  UTIL WORK THREAD SUPPORT - - - - - - - - - - - - 

void util_thread_create(
    void * (*proc)(void * cx),
    void * cx)
{
    sigset_t  set, oldset;
    pthread_t thread_handle;
    int32_t ret;

    // mask all signals
    sigfillset(&set);
    pthread_sigmask(SIG_SETMASK, &set, &oldset);

    // create the thread
    ret = pthread_create(&thread_handle, NULL, proc, cx);
    ASSERT_MSG("ret == 0", "%s", strerror(ret));

    // restore signal mask
    pthread_sigmask(SIG_SETMASK, &oldset, NULL);
}

void * util_work_thread(
    void * cx_arg)
{
    while (TRUE) {
        // barrier
        pthread_barrier_wait(&wt_barrier);

        // perform requested operation
        switch (wt_req.req) {
        case WORK_THREAD_REQ_GIVE_TEST: {
            uint32_t org_idx, img_idx;

            while (TRUE) {
                org_idx = __sync_fetch_and_add(&wt_req.u.give_test.org_idx, 1);
                if (org_idx >= score->max_org || ctrl_c) {
                    break;
                }

                for (img_idx = 0; img_idx < score->max_image_tbl; img_idx++) {
                    if (ctrl_c) {
                        break;
                    }

                    score->org_answer[org_idx][img_idx].ans_nn =
                            nn_eval(org_idx, score->image_tbl[img_idx].pixels); 
                }
            }
            break; }

        case WORK_THREAD_REQ_SCORE_TEST: {
            uint32_t org_idx, img_idx;

            while (TRUE) {
                org_idx = __sync_fetch_and_add(&wt_req.u.score_test.org_idx, 1);
                if (org_idx >= score->max_org || ctrl_c) {
                    break;
                }

                for (img_idx = 0; img_idx < score->max_image_tbl; img_idx++) {
                    uint32_t best_ans, best_ans_incorrect_neuron_count;
                    score_org_ans_t * soa = &score->org_answer[org_idx][img_idx];

                    // check for ctrl_c
                    if (ctrl_c) {
                        break;
                    }

                    // find the fbest answers and their associated number of incorrect neurons
                    ga_test_score_find_best_answer(soa->ans_nn,
                                                   &best_ans, 
                                                   &best_ans_incorrect_neuron_count);

                    // set the ans_status
                    if (best_ans == -1 || best_ans_incorrect_neuron_count > MAX_INCORRECT_NEURON_CNT) {
                        soa->ans_status = ANSWER_IS_DONT_KNOW;
                    } else if (strcmp(pop->answer[best_ans].name, score->image_tbl[img_idx].ans_name) != 0) {
                        soa->ans_status = ANSWER_IS_INCORRECT;
                    } else {
                        soa->ans_status = ANSWER_IS_CORRECT;
                    }

                    // save the best_ans_incorrect_neuron_count
                    soa->incorrect_neuron_count = best_ans_incorrect_neuron_count;
                }
            }
            break; }

        default:
            ASSERT_MSG(0,NULL);
            break;
        }

        // barrier
        pthread_barrier_wait(&wt_barrier);
    }

    return NULL;
}

// - - - - - - - - -  UTIL HASH TABLE SUPPORT  - - - - - - - - - - - - 

uint32_t util_hash_lookup_ans_name(   // returns ans or -1 if not found
    char * ans_name)
{
    uint32_t idx, i;

    idx = crc32(0,(void*)ans_name,strlen(ans_name)) % MAX_ANSWER_HASH_TBL;

    for (i = 0; i < MAX_ANSWER_HASH_TBL; i++) {
        if (pop->answer_hash_tbl[idx] == -1) {
            return -1;
        }
        if (strcmp(pop->answer[pop->answer_hash_tbl[idx]].name, ans_name) == 0) {
            return pop->answer_hash_tbl[idx];
        }
        idx = (idx + 1) % MAX_ANSWER_HASH_TBL;
    }

    PRINT_ERROR("searched the entire hash_tbl\n");
    return -1;
}

void util_hash_add_ans_name(
    char * ans_name,
    uint32_t ans)
{
    uint32_t idx,i;

    idx = crc32(0,(void*)ans_name,strlen(ans_name)) % MAX_ANSWER_HASH_TBL;

    for (i = 0; i < MAX_ANSWER_HASH_TBL; i++) {
        if (pop->answer_hash_tbl[idx] == -1) {
            pop->answer_hash_tbl[idx] = ans;
            return;
        }
        idx = (idx + 1) % MAX_ANSWER_HASH_TBL;
    }

    PRINT_ERROR("failed add ans_name %s to hash_tbl\n", ans_name);
}

uint32_t util_hash_lookup_image(   // returns img; or -1 for not found
    image_t * image)
{
    uint32_t idx, i;

    idx = crc32(0,(void*)image->pixels,MAX_PIXELS) % MAX_IMAGE_HASH_TBL;

    for (i = 0; i < MAX_IMAGE_HASH_TBL; i++) {
        if (pop->image_hash_tbl[idx] == -1) {
            return -1;
        }
        if (memcmp(image->pixels, pop->image_tbl[pop->image_hash_tbl[idx]].pixels, MAX_PIXELS) == 0) {
            return pop->image_hash_tbl[idx];
        }
        idx = (idx + 1) % MAX_IMAGE_HASH_TBL;
    }

    PRINT_ERROR("searched the entire hash_tbl\n");
    return -1;
}

void util_hash_add_image(
    image_t * image,
    uint32_t img)
{
    uint32_t idx,i;

    idx = crc32(0,(void*)image->pixels,MAX_PIXELS) % MAX_IMAGE_HASH_TBL;

    for (i = 0; i < MAX_IMAGE_HASH_TBL; i++) {
        if (pop->image_hash_tbl[idx] == -1) {
            pop->image_hash_tbl[idx] = img;
            return;
        }
        idx = (idx + 1) % MAX_IMAGE_HASH_TBL;
    }

    PRINT_ERROR("failed add image %s to hash_tbl\n", image->filename);
}

// - - - - - - - - -  UTIL GENERAL SUPPORT - - - - - - - - - - - - - - 

void util_ctrl_c_init(
    void)
{
    struct sigaction sa;
    int32_t ret;

    bzero(&sa,sizeof(sa));
    sa.sa_handler = util_ctrl_c_hndlr;
    sa.sa_flags = SA_RESTART;
    ret = sigaction(SIGINT, &sa, NULL);
    if (ret < 0) {
        PRINT_ERROR("sigaction, %s\n", strerror(errno));
    }
}

void util_ctrl_c_hndlr(
    int32_t sig)
{
    ctrl_c = TRUE;
}

void util_assert(
        const char *expression, const char *func, const char *file,
        uint32_t line, char * fmt, ...)
{
    va_list ap;
    char msg[1000];

    // sprint msg 
    msg[0] = '\0';
    if (msg != NULL) {
        va_start(ap, fmt);
        vsprintf(msg, fmt, ap);
        va_end(ap);
    }

    // print
    printf("ASSERTION FAILED in %s\n"
           "  expr      : %s\n"
           "  file,line : %s %d\n"
           "  message   : %s\n",
           func, expression, file, line, msg);

    // exit
    exit(1);
}

void util_timing(
    timing_t * timing,
    uint32_t idx)
{
    uint64_t curr, interval;

    curr = util_get_time_ns();

    switch (idx) {
    case TIMING_INIT:
        bzero(timing, sizeof(timing_t));
        break;
    case TIMING_IGNORE:
        break;
    default:
        interval = curr - timing->last;
        timing->stat[idx] += interval;
        timing->total += interval;
    }

    timing->last = curr;
}

uint64_t util_get_time_ns(
    void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

void util_pop_log(
    char * fmt, ...)
{
    va_list ap;
    char fmt_msg[1000], full_msg[1000];
    struct timeval tv;
    struct tm tm;

    va_start(ap, fmt);
    vsprintf(fmt_msg, fmt, ap);
    va_end(ap);

    gettimeofday(&tv, NULL);
    localtime_r(&tv.tv_sec, &tm);

    sprintf(full_msg, 
            "%2.2d/%2.2d %2.2d:%2.2d - %d %d %d - %d %d %d %d %d %d %d %s - %s\n",
            tm.tm_mon+1, tm.tm_mday, tm.tm_hour, tm.tm_min, 
            pop->gen_num,
            pop->max_image_tbl,
            pop->param_max_org,
            pop->neuron_lvl,
            pop->param_max_neuron[0],
            pop->param_max_neuron[1],
            pop->param_max_neuron[2],
            pop->param_max_incorrect_neuron_cnt[0],
            pop->param_max_incorrect_neuron_cnt[1],
            pop->param_max_incorrect_neuron_cnt[2],
            pop->param_image_filename,
            fmt_msg);

    printf("%s", full_msg);

    size_t len = strlen(full_msg);
    if (pop->max_log + len > MAX_POP_LOG-1) {
        PRINT_ERROR("pop log is full\n");
        return;
    }

    memcpy(pop->log + pop->max_log, full_msg, len);
    pop->max_log += len;
}

char * util_ans_nn_to_str(
    ans_nn_t val,
    uint32_t base)
{
    // note - max base 10 is 340282366920938463463374607431768211455 (39 digits)

    #define MAX_STR_POOL            4
    #define MAX_UINT128_STR_BASE_10 40

    int32_t         i, n=0;
    char            digits[MAX_UINT128_STR_BASE_10];
    char          * str;
    static uint32_t str_pool_idx;
    static char     str_pool[4][MAX_UINT128_STR_BASE_10];

    str = str_pool[str_pool_idx++ % MAX_STR_POOL];

    if (base != 10 && base != 16) {
        strcpy(str, "invalid-base");
        return str;
    }

    do {
        digits[n++] = "0123456789abcdef"[val%base];
        val /= base;
    } while (val);

    for (i = 0; i < n; i++) {
        str[i] = digits[n-i-1];
    }
    str[i] = '\0';

    return str;
}

void util_parse_input_line(
    char * input_line,
    char ** cmd,
    char ** args)
{
    size_t len, i;

    // preset return values
    *cmd = "";
    *args = "";

    // remove newline char that may be at the end of input_line
    len = strlen(input_line);
    if (len > 0 && input_line[len-1] == '\n') {
        input_line[len-1] = '\0';
    }

    // remove any leading spaces in input_line;
    // if nothing remains then return
    while (*input_line == ' ') {
        input_line++;
    }
    if (*input_line == '\0') {
        return;
    }

    // set command to input_line, null terminate, and return if no args follow
    *cmd = input_line;
    while (*input_line != ' ' && *input_line != '\0') {
        input_line++;
    }
    if (*input_line == '\0') {
        return;
    }
    *input_line = '\0';
    input_line++;

    // find the begining of args, if no args then return
    while (*input_line == ' ') {
        input_line++;
    }
    if (*input_line == '\0') {
        return;
    }

    // set args, and remove trailing spaces in args
    *args = input_line;
    for (i = strlen(input_line)-1; i > 0; i--) {
        if (input_line[i] != ' ') {
            break;
        }
        input_line[i] = '\0';
    }
}

uint32_t num_bits_diff(
    __uint128_t a, 
    __uint128_t b)
{
    __uint128_t xor       = a ^ b;
    uint64_t    xor_low   = (uint64_t)xor;
    uint64_t    xor_high  = (uint64_t)(xor>>64);
    uint32_t    bits_diff = 0;

    if (xor_low) {
        bits_diff += util_popcount_wikipedia(xor_low);
    }

    if (xor_high) {
        bits_diff += util_popcount_wikipedia(xor_high);
    }

    return bits_diff;
}


// timing comparison for these 2 popcount routines doing popcount
// on all integers in range 0 - 4000000000 is:
// - util_popcount_x86       3.445 secs
// - util_popcount_wikipedia 2.784 secs

uint32_t util_popcount_x86(
    uint64_t x)
{
    uint64_t ret;

    // note - this is supported on recent x86 only
    asm("popcntq %1, %0;":"=r"(ret) :"r"(x));
    return ret;
}

const uint64_t m1  = 0x5555555555555555; //binary: 0101...
const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
const uint64_t m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
const uint64_t m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
const uint64_t m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
const uint64_t hff = 0xffffffffffffffff; //binary: all ones
const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...

//This uses fewer arithmetic operations than any other known  
//implementation on machines with fast multiplication.
//It uses 12 arithmetic operations, one of which is a multiply.
uint32_t util_popcount_wikipedia(
    uint64_t x)
{
    x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
    x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits 
    x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits 
    return (x * h01)>>56;  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
}

void util_swap(
    void * x, 
    void * y, 
    size_t len)
{
    size_t len_remaining, xfer_len;
    char buff[10000];

    for (len_remaining = len; len_remaining; ) {
        xfer_len = (len_remaining > sizeof(buff) ? sizeof(buff) : len_remaining);

        memcpy(buff,x,xfer_len);
        memcpy(x,y,xfer_len);
        memcpy(y,buff,xfer_len);

        x += xfer_len;
        y += xfer_len;
        len_remaining -= xfer_len;
    }
}

