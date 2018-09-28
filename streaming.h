#include <stdio.h>
#include <vector>

#include <iostream>
#include <fstream>
#include <time.h>
#define _XOPEN_SOURCE 600 /* for usleep */

extern "C"
{
    #ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
//#include <libavutil/timestamp.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfiltergraph.h>
#include <libavfilter/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>



#include <math.h>
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/common.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/samplefmt.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfiltergraph.h>
#include <libavfilter/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>


#include <libavutil/opt.h>
#include <libavutil/mathematics.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
}

// OpenCV
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif



using namespace std;

#pragma comment(lib, "dev\\avcodec.lib")
#pragma comment(lib, "dev\\avformat.lib")
#pragma comment(lib, "dev\\avfilter.lib")

//#pragma comment(lib, "dev\\avcodec.lib")
#pragma comment(lib, "dev\\avdevice.lib")
//#pragma comment(lib, "dev\\avfilter.lib")
//#pragma comment(lib, "dev\\avformat.lib")
#pragma comment(lib, "dev\\avutil.lib")
#pragma comment(lib, "dev\\postproc.lib")
#pragma comment(lib, "dev\\swresample.lib")
#pragma comment(lib, "dev\\swscale.lib")


static int audio_is_eof, video_is_eof;

#define STREAM_DURATION   50.0
#define STREAM_FRAME_RATE 25 /* 25 images/s */
#define STREAM_PIX_FMT    AV_PIX_FMT_YUV420P /* default pix_fmt */

static int sws_flags = SWS_BICUBIC;

static void log_packet(const AVFormatContext *fmt_ctx, const AVPacket *pkt)
{
    AVRational *time_base = &fmt_ctx->streams[pkt->stream_index]->time_base;

    /*printf("pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s stream_index:%d\n",
           av_ts_make_string(pkt->pts), av_ts_make_string(pkt->pts, time_base),
           av_ts_make_string(pkt->dts), av_ts_make_string(pkt->dts, time_base),
           av_ts_make_string(pkt->duration), av_ts_make_string(pkt->duration, time_base),
           pkt->stream_index);*/
}

static int write_frame(AVFormatContext *fmt_ctx, const AVRational *time_base, AVStream *st, AVPacket *pkt)
{
    /* rescale output packet timestamp values from codec to stream timebase */
    pkt->pts = av_rescale_q_rnd(pkt->pts, *time_base, st->time_base, AVRounding(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
    pkt->dts = av_rescale_q_rnd(pkt->dts, *time_base, st->time_base, AVRounding(AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX));
    pkt->duration = av_rescale_q(pkt->duration, *time_base, st->time_base);
    pkt->stream_index = st->index;

    /* Write the compressed frame to the media file. */
    log_packet(fmt_ctx, pkt);
    return av_interleaved_write_frame(fmt_ctx, pkt);
}

/* Add an output stream. */
static AVStream *add_stream(AVFormatContext *oc, AVCodec **codec,
                            enum AVCodecID codec_id)
{
    AVCodecContext *c;
    AVStream *st;

    /* find the encoder */
    *codec = avcodec_find_encoder(codec_id);
    if (!(*codec)) {
        fprintf(stderr, "Could not find encoder for '%s'\n",
                avcodec_get_name(codec_id));
        exit(1);
    }

    st = avformat_new_stream(oc, *codec);
    if (!st) {
        fprintf(stderr, "Could not allocate stream\n");
        exit(1);
    }
    st->id = oc->nb_streams-1;
    c = st->codec;

    switch ((*codec)->type) {
    case AVMEDIA_TYPE_AUDIO:
        c->sample_fmt  = (*codec)->sample_fmts ?
            (*codec)->sample_fmts[0] : AV_SAMPLE_FMT_FLTP;
        c->bit_rate    = 64000;
        c->sample_rate = 44100;
        c->channels    = 2;
        break;

    case AVMEDIA_TYPE_VIDEO:
        c->codec_id = codec_id;
        if(codec_id == CODEC_ID_H264)
            cout<<"Codec ID  "<<(AVCodecID)codec_id<<endl;
        c->bit_rate = 400000;
        /* Resolution must be a multiple of two. */
        c->width    = 352;
        c->height   = 288;
        /* timebase: This is the fundamental unit of time (in seconds) in terms
         * of which frame timestamps are represented. For fixed-fps content,
         * timebase should be 1/framerate and timestamp increments should be
         * identical to 1. */
        c->time_base.den = STREAM_FRAME_RATE;
        c->time_base.num = 1;
        c->gop_size      = 12; /* emit one intra frame every twelve frames at most */
        c->pix_fmt       = STREAM_PIX_FMT;
        if (c->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
            /* just for testing, we also add B frames */
            c->max_b_frames = 2;
        }
        if (c->codec_id == AV_CODEC_ID_MPEG1VIDEO) {
            /* Needed to avoid using macroblocks in which some coeffs overflow.
             * This does not happen with normal video, it just happens here as
             * the motion of the chroma plane does not match the luma plane. */
            c->mb_decision = 2;
        }
    break;

    default:
        break;
    }

    /* Some formats want stream headers to be separate. */
    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;

    return st;
}

/**************************************************************/
/* audio output */

static float t, tincr, tincr2;

AVFrame *audio_frame;
static uint8_t **src_samples_data;
static int       src_samples_linesize;
static int       src_nb_samples;

static int max_dst_nb_samples;
uint8_t **dst_samples_data;
int       dst_samples_linesize;
int       dst_samples_size;
int samples_count;

struct SwrContext *swr_ctx = NULL;

static void open_audio(AVFormatContext *oc, AVCodec *codec, AVStream *st)
{
    AVCodecContext *c;
    int ret;

    c = st->codec;

    /* allocate and init a re-usable frame */
    audio_frame = av_frame_alloc();
    if (!audio_frame) {
        fprintf(stderr, "Could not allocate audio frame\n");
        exit(1);
    }


    //The encoder 'aac' is experimental but experimental codecs are not enabled,
    //  add '-strict -2' if you want to use it
    c->strict_std_compliance = -2;

    /* open it */
    ret = avcodec_open2(c, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "Could not open audio codec:\n");
        exit(1);
    }

    /* init signal generator */
    t     = 0;
    tincr = 2 * M_PI * 110.0 / c->sample_rate;
    /* increment frequency by 110 Hz per second */
    tincr2 = 2 * M_PI * 110.0 / c->sample_rate / c->sample_rate;

    src_nb_samples = c->codec->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE ?
        10000 : c->frame_size;

    ret = av_samples_alloc_array_and_samples(&src_samples_data, &src_samples_linesize, c->channels,
                                             src_nb_samples, AV_SAMPLE_FMT_S16, 0);
    if (ret < 0) {
        fprintf(stderr, "Could not allocate source samples\n");
        exit(1);
    }

    /* compute the number of converted samples: buffering is avoided
     * ensuring that the output buffer will contain at least all the
     * converted input samples */
    max_dst_nb_samples = src_nb_samples;

    /* create resampler context */
    if (c->sample_fmt != AV_SAMPLE_FMT_S16) {
        swr_ctx = swr_alloc();
        if (!swr_ctx) {
            fprintf(stderr, "Could not allocate resampler context\n");
            exit(1);
        }

        /* set options */
        av_opt_set_int       (swr_ctx, "in_channel_count",   c->channels,       0);
        av_opt_set_int       (swr_ctx, "in_sample_rate",     c->sample_rate,    0);
        av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt",      AV_SAMPLE_FMT_S16, 0);
        av_opt_set_int       (swr_ctx, "out_channel_count",  c->channels,       0);
        av_opt_set_int       (swr_ctx, "out_sample_rate",    c->sample_rate,    0);
        av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt",     c->sample_fmt,     0);

        /* initialize the resampling context */
        if ((ret = swr_init(swr_ctx)) < 0) {
            fprintf(stderr, "Failed to initialize the resampling context\n");
            exit(1);
        }

        ret = av_samples_alloc_array_and_samples(&dst_samples_data, &dst_samples_linesize, c->channels,
                                                 max_dst_nb_samples, c->sample_fmt, 0);
        if (ret < 0) {
            fprintf(stderr, "Could not allocate destination samples\n");
            exit(1);
        }
    } else {
        dst_samples_data = src_samples_data;
    }
    dst_samples_size = av_samples_get_buffer_size(NULL, c->channels, max_dst_nb_samples,
                                                  c->sample_fmt, 0);
}


/**************************************************************/
/* video output */

static AVPicture src_picture, dst_picture;
static int frame_count;

static void open_video(AVFormatContext *oc, AVCodec *codec, AVStream *st, AVFrame *frame)
{
    int ret;
    AVCodecContext *c = st->codec;

    /* open the codec */
    ret = avcodec_open2(c, codec, NULL);
    if (ret < 0) {
        fprintf(stderr, "Could not open video codec: ");
        exit(1);
    }

    /* allocate and init a re-usable frame */

    //#frame = av_frame_alloc();

    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }
    frame->format = c->pix_fmt;
    //frame->width = c->width;
    //frame->height = c->height;

    /* Allocate the encoded raw picture. */
    ret = avpicture_alloc(&dst_picture, c->pix_fmt, c->width, c->height);
    if (ret < 0) {
        fprintf(stderr, "Could not allocate picture: ");
        exit(1);
    }

    /* If the output format is not YUV420P, then a temporary YUV420P
     * picture is needed too. It is then converted to the required
     * output format. */
    if (c->pix_fmt != AV_PIX_FMT_YUV420P) {
        ret = avpicture_alloc(&src_picture, AV_PIX_FMT_YUV420P, c->width, c->height);
        if (ret < 0) {
            fprintf(stderr, "Could not allocate temporary picture:");
            exit(1);
        }
    }

    /* copy data and linesize picture pointers to frame */
    *((AVPicture *)frame) = dst_picture;
}

/* Prepare a dummy image. */
static void fill_yuv_image(AVPicture *pict, int frame_index,
                           int width, int height)
{
    int x, y, i;

    i = frame_index;

    /* Y */
    for (y = 0; y < height; y++)
        for (x = 0; x < width; x++)
            pict->data[0][y * pict->linesize[0] + x] = x + y + i * 3;

    /* Cb and Cr */
    for (y = 0; y < height / 2; y++) {
        for (x = 0; x < width / 2; x++) {
            pict->data[1][y * pict->linesize[1] + x] = 128 + y + i * 2;
            pict->data[2][y * pict->linesize[2] + x] = 64 + x + i * 5;
        }
    }
}

static void write_video_frame(AVFormatContext *oc, AVStream *st, int flush, AVFrame* frame)
{
    int ret;
    static struct SwsContext *sws_ctx;
    AVCodecContext *c = st->codec;

    if (!flush) {
        /* as we only generate a YUV420P picture, we must convert it
         * to the codec pixel format if needed */
        if (!sws_ctx) {
            sws_ctx = sws_getContext(c->width, c->height, AV_PIX_FMT_YUV420P,
                                     c->width, c->height, c->pix_fmt,
                                     sws_flags, NULL, NULL, NULL);
            if (!sws_ctx) {
                fprintf(stderr,
                        "Could not initialize the conversion context\n");
                exit(1);
            }
        }
        //fill_yuv_image(&src_picture, frame_count, c->width, c->height);
        sws_scale(sws_ctx,
                  (const uint8_t * const *)frame->data, frame->linesize,
                  0, c->height, dst_picture.data, dst_picture.linesize);
    }

    if (oc->oformat->flags & AVFMT_RAWPICTURE && !flush) {
        std::cout<<"/* Raw video case - directly store the picture in the packet */" <<std::endl;
        /* Raw video case - directly store the picture in the packet */
        AVPacket pkt;
        av_init_packet(&pkt);

        pkt.flags        |= AV_PKT_FLAG_KEY;
        pkt.stream_index  = st->index;
        pkt.data          = dst_picture.data[0];
        pkt.size          = sizeof(AVPicture);

        ret = av_interleaved_write_frame(oc, &pkt);
    } else {
        AVPacket pkt = { 0 };
        int got_packet;
        av_init_packet(&pkt);

        /* encode the image */
        frame->pts = frame_count;
        ret = avcodec_encode_video2(c, &pkt, flush ? NULL : frame, &got_packet);
        if (ret < 0) {
            fprintf(stderr, "Error encoding video frame:");
            exit(1);
        }
        /* If size is zero, it means the image was buffered. */

        if (got_packet) {
            //cout<<"got Packet"<<endl;
            ret = write_frame(oc, &c->time_base, st, &pkt);
        } else {
            //cout<<"EOF\n";
            if (flush)
                video_is_eof = 1;
            ret = 0;
        }
    }

    if (ret < 0) {
        fprintf(stderr, "Error while writing video frame: ");
        exit(1);
    }
    frame_count++;
}

static void close_video(AVFormatContext *oc, AVStream *st, AVFrame* frame)
{
    avcodec_close(st->codec);
    av_free(src_picture.data[0]);
    av_free(dst_picture.data[0]);
    av_frame_free(&frame);
}






