#include "profiler.h"

#ifdef NPROFILER

void profiler_reset(){}
void profiler_segments_print(long flops16, long flops32, long flops64)
{
    //dummy statements to avoid warnings
    (void)flops16;
    (void)flops32;
    (void)flops64;
}

#else //NPROFILER

#include <stdio.h>
#include <string.h>

#include "machine.h"

struct profile_segment *profile_segment_list;
struct profile_segment profile_segment_sentinel;
struct profile_segment *profile_segment_current = &profile_segment_sentinel;
char json_output[1024];

void profiler_reset()
{
    struct profile_segment *next;
    for(struct profile_segment *segment = profile_segment_list;
        segment;
        segment = next)
    {
        next = segment->next;
        *segment = (struct profile_segment){.name = segment->name};
    }
    profile_segment_list = NULL;
    profile_segment_sentinel = (struct profile_segment){0};
    profile_segment_current = &profile_segment_sentinel;
    memset(json_output, 0, sizeof(json_output));
}

static void profiler_segments_print_children_json(struct profile_segment *parent, int level)
{
    for(struct profile_segment *segment = profile_segment_list;
        segment;
        segment = segment->next)
    {
        if(segment->parent != parent)
            continue;

        double totalTime = (double)segment->totalTime/1e9;
        double childTime = (double)segment->childTime/1e9;
        double selfTime = totalTime - childTime;
        char buffer[32];
        // We only care about segments at level 1
        if (level == 1) {
            snprintf(buffer, sizeof(buffer), "%s,%lf,", segment->name, selfTime);
            strcat(json_output, buffer);
        }

        profiler_segments_print_children_json(segment, level+1);
    }
}

static void profiler_segments_print_children(struct profile_segment *parent, double parentTime, int level)
{
    double overallTime = (double)profile_segment_sentinel.childTime/1e9;
    static const char *spaces = "                                    ";
    for(struct profile_segment *segment = profile_segment_list;
        segment;
        segment = segment->next)
    {
        if(segment->parent != parent)
            continue;

        double totalTime = (double)segment->totalTime/1e9;
        double childTime = (double)segment->childTime/1e9;
        double selfTime = totalTime - childTime;
        double absPercent = totalTime / overallTime * 100;
        double relativePercent = totalTime / parentTime * 100;
        printf("%*.s%s%*.s %6.2f%% %*.s %fs %6.2f%%  %fs\n", level * 2, spaces,
                segment->name, 20 - (int)strlen(segment->name), spaces,
                relativePercent, 12 - level * 2, spaces,
                totalTime, absPercent, selfTime);

        profiler_segments_print_children(segment, totalTime, level+1);
    }
}

static void profiler_reverse_list()
{
    struct profile_segment *new_list = NULL;
    struct profile_segment *next;
    for(struct profile_segment *segment = profile_segment_list;
        segment;
        segment = next)
    {
        next = segment->next;
        segment->next = new_list;
        new_list = segment;
    }
    profile_segment_list = new_list;
}

void profiler_segments_print(long flops16, long flops32, long flops64)
{
    double overallTime = (double)profile_segment_sentinel.childTime / 1e9; // convert from ns to seconds
    
    // convert from flop/s to Gflop/s
    if (flops16 > 0 || flops32 > 0 || flops64 > 0) {
        if (flops16 > 0) printf("FP16: %.2f Gflop/s ", (double)flops16 / overallTime / 1e9);
        if (flops32 > 0) printf("FP32: %.2f Gflop/s ", (double)flops32 / overallTime / 1e9);
        if (flops64 > 0) printf("FP64: %.2f Gflop/s ", (double)flops64 / overallTime / 1e9);
        printf("\n");
    }

    printf("%-20s %-20s %-9s %-8s %-9s\n", "name", "relative %", "total", "total %", "self");
    profiler_reverse_list();
    profiler_segments_print_children(&profile_segment_sentinel, overallTime, 0);
}

char* profiler_segments_print_json() {
    profiler_reverse_list();
    profiler_segments_print_children_json(&profile_segment_sentinel, 0);
    return &json_output[0];
}

extern inline void profile_segment_start(struct profile_segment *segment);
extern inline void profile_segments_start(struct profile_segment *segment0, 
                                   struct profile_segment *segment1);
extern inline void profile_segments_switch(struct profile_segment *newSegment);
extern inline void profile_segment_end();
extern inline void profile_segments_end();

#endif //NPROFILER
