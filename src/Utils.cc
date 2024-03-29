/*
    This file is part of ElephantArt.
    Copyright (C) 2021 Hung-Zhe Lin

    ElephantArt is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElephantArt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElephantArt.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <mutex>

#include "Utils.h"
#include "config.h"

namespace Utils {

static constexpr auto z_entries = 1000;
static constexpr float z_lookup[z_entries] = {
31830.9902343750f, 223.6034393311f, 47.9277305603f, 23.3321819305f, 15.5468549728f, 12.0316534042f, 10.1026840210f, 8.9070272446f, 8.1020574570f, 7.5269541740f,
7.0973553658f, 6.7651691437f, 6.5011448860f, 6.2865495682f, 6.1088681221f, 5.9594416618f, 5.8320994377f, 5.7223310471f, 5.6267662048f, 5.5428385735f,
5.4685611725f, 5.4023718834f, 5.3430266380f, 5.2895226479f, 5.2410430908f, 5.1969156265f, 5.1565823555f, 5.1195764542f, 5.0855045319f, 5.0540323257f,
5.0248742104f, 4.9977855682f, 4.9725537300f, 4.9489946365f, 4.9269485474f, 4.9062743187f, 4.8868479729f, 4.8685603142f, 4.8513145447f, 4.8350238800f,
4.8196115494f, 4.8050084114f, 4.7911529541f, 4.7779889107f, 4.7654657364f, 4.7535381317f, 4.7421646118f, 4.7313075066f, 4.7209324837f, 4.7110080719f,
4.7015056610f, 4.6923990250f, 4.6836643219f, 4.6752781868f, 4.6672210693f, 4.6594738960f, 4.6520190239f, 4.6448402405f, 4.6379227638f, 4.6312518120f,
4.6248154640f, 4.6186008453f, 4.6125969887f, 4.6067934036f, 4.6011800766f, 4.5957479477f, 4.5904884338f, 4.5853934288f, 4.5804548264f, 4.5756664276f,
4.5710210800f, 4.5665121078f, 4.5621337891f, 4.5578813553f, 4.5537481308f, 4.5497303009f, 4.5458221436f, 4.5420198441f, 4.5383191109f, 4.5347161293f,
4.5312061310f, 4.5277867317f, 4.5244541168f, 4.5212044716f, 4.5180354118f, 4.5149431229f, 4.5119261742f, 4.5089807510f, 4.5061049461f, 4.5032958984f,
4.5005512238f, 4.4978694916f, 4.4952478409f, 4.4926843643f, 4.4901776314f, 4.4877252579f, 4.4853253365f, 4.4829769135f, 4.4806780815f, 4.4784269333f,
4.4762225151f, 4.4740629196f, 4.4719467163f, 4.4698734283f, 4.4678411484f, 4.4658484459f, 4.4638943672f, 4.4619784355f, 4.4600987434f, 4.4582543373f,
4.4564447403f, 4.4546689987f, 4.4529252052f, 4.4512133598f, 4.4495325089f, 4.4478816986f, 4.4462604523f, 4.4446673393f, 4.4431018829f, 4.4415636063f,
4.4400515556f, 4.4385652542f, 4.4371037483f, 4.4356665611f, 4.4342536926f, 4.4328632355f, 4.4314961433f, 4.4301505089f, 4.4288268089f, 4.4275240898f,
4.4262418747f, 4.4249796867f, 4.4237375259f, 4.4225139618f, 4.4213089943f, 4.4201226234f, 4.4189538956f, 4.4178028107f, 4.4166688919f, 4.4155516624f,
4.4144506454f, 4.4133653641f, 4.4122958183f, 4.4112420082f, 4.4102025032f, 4.4091782570f, 4.4081678391f, 4.4071717262f, 4.4061894417f, 4.4052205086f,
4.4042644501f, 4.4033217430f, 4.4023919106f, 4.4014739990f, 4.4005684853f, 4.3996748924f, 4.3987927437f, 4.3979225159f, 4.3970632553f, 4.3962149620f,
4.3953776360f, 4.3945508003f, 4.3937344551f, 4.3929281235f, 4.3921322823f, 4.3913459778f, 4.3905692101f, 4.3898019791f, 4.3890442848f, 4.3882956505f,
4.3875555992f, 4.3868250847f, 4.3861026764f, 4.3853888512f, 4.3846836090f, 4.3839859962f, 4.3832969666f, 4.3826160431f, 4.3819422722f, 4.3812766075f,
4.3806185722f, 4.3799676895f, 4.3793239594f, 4.3786878586f, 4.3780584335f, 4.3774361610f, 4.3768205643f, 4.3762116432f, 4.3756093979f, 4.3750133514f,
4.3744239807f, 4.3738408089f, 4.3732638359f, 4.3726925850f, 4.3721280098f, 4.3715686798f, 4.3710155487f, 4.3704676628f, 4.3699259758f, 4.3693895340f,
4.3688588142f, 4.3683328629f, 4.3678126335f, 4.3672976494f, 4.3667879105f, 4.3662829399f, 4.3657827377f, 4.3652877808f, 4.3647975922f, 4.3643121719f,
4.3638315201f, 4.3633551598f, 4.3628840446f, 4.3624167442f, 4.3619542122f, 4.3614959717f, 4.3610420227f, 4.3605923653f, 4.3601465225f, 4.3597054482f,
4.3592681885f, 4.3588347435f, 4.3584055901f, 4.3579797745f, 4.3575582504f, 4.3571405411f, 4.3567266464f, 4.3563160896f, 4.3559093475f, 4.3555064201f,
4.3551068306f, 4.3547110558f, 4.3543186188f, 4.3539290428f, 4.3535432816f, 4.3531608582f, 4.3527817726f, 4.3524060249f, 4.3520331383f, 4.3516635895f,
4.3512973785f, 4.3509340286f, 4.3505735397f, 4.3502163887f, 4.3498620987f, 4.3495106697f, 4.3491621017f, 4.3488163948f, 4.3484735489f, 4.3481335640f,
4.3477964401f, 4.3474617004f, 4.3471298218f, 4.3468008041f, 4.3464741707f, 4.3461499214f, 4.3458285332f, 4.3455095291f, 4.3451933861f, 4.3448796272f,
4.3445677757f, 4.3442587852f, 4.3439521790f, 4.3436479568f, 4.3433461189f, 4.3430461884f, 4.3427486420f, 4.3424539566f, 4.3421607018f, 4.3418703079f,
4.3415818214f, 4.3412952423f, 4.3410110474f, 4.3407287598f, 4.3404488564f, 4.3401708603f, 4.3398947716f, 4.3396210670f, 4.3393487930f, 4.3390789032f,
4.3388109207f, 4.3385448456f, 4.3382806778f, 4.3380184174f, 4.3377580643f, 4.3374996185f, 4.3372426033f, 4.3369879723f, 4.3367347717f, 4.3364830017f,
4.3362336159f, 4.3359856606f, 4.3357396126f, 4.3354949951f, 4.3352522850f, 4.3350110054f, 4.3347716331f, 4.3345336914f, 4.3342976570f, 4.3340630531f,
4.3338298798f, 4.3335981369f, 4.3333683014f, 4.3331398964f, 4.3329129219f, 4.3326878548f, 4.3324637413f, 4.3322415352f, 4.3320202827f, 4.3318009377f,
4.3315830231f, 4.3313660622f, 4.3311510086f, 4.3309369087f, 4.3307247162f, 4.3305134773f, 4.3303036690f, 4.3300952911f, 4.3298878670f, 4.3296823502f,
4.3294777870f, 4.3292741776f, 4.3290724754f, 4.3288717270f, 4.3286724091f, 4.3284740448f, 4.3282771111f, 4.3280811310f, 4.3278865814f, 4.3276934624f,
4.3275012970f, 4.3273100853f, 4.3271203041f, 4.3269319534f, 4.3267440796f, 4.3265576363f, 4.3263726234f, 4.3261880875f, 4.3260049820f, 4.3258233070f,
4.3256421089f, 4.3254623413f, 4.3252835274f, 4.3251061440f, 4.3249292374f, 4.3247537613f, 4.3245787621f, 4.3244051933f, 4.3242325783f, 4.3240609169f,
4.3238906860f, 4.3237209320f, 4.3235521317f, 4.3233842850f, 4.3232178688f, 4.3230519295f, 4.3228869438f, 4.3227229118f, 4.3225603104f, 4.3223981857f,
4.3222370148f, 4.3220767975f, 4.3219170570f, 4.3217587471f, 4.3216009140f, 4.3214445114f, 4.3212885857f, 4.3211336136f, 4.3209791183f, 4.3208260536f,
4.3206734657f, 4.3205218315f, 4.3203711510f, 4.3202209473f, 4.3200716972f, 4.3199234009f, 4.3197755814f, 4.3196287155f, 4.3194828033f, 4.3193373680f,
4.3191928864f, 4.3190493584f, 4.3189063072f, 4.3187642097f, 4.3186225891f, 4.3184819221f, 4.3183417320f, 4.3182024956f, 4.3180642128f, 4.3179264069f,
4.3177890778f, 4.3176527023f, 4.3175168037f, 4.3173818588f, 4.3172478676f, 4.3171138763f, 4.3169808388f, 4.3168487549f, 4.3167171478f, 4.3165860176f,
4.3164558411f, 4.3163261414f, 4.3161973953f, 4.3160691261f, 4.3159413338f, 4.3158140182f, 4.3156876564f, 4.3155622482f, 4.3154368401f, 4.3153123856f,
4.3151884079f, 4.3150649071f, 4.3149423599f, 4.3148202896f, 4.3146986961f, 4.3145775795f, 4.3144574165f, 4.3143372536f, 4.3142180443f, 4.3140997887f,
4.3139815331f, 4.3138642311f, 4.3137469292f, 4.3136305809f, 4.3135147095f, 4.3133997917f, 4.3132848740f, 4.3131709099f, 4.3130569458f, 4.3129439354f,
4.3128314018f, 4.3127193451f, 4.3126077652f, 4.3124966621f, 4.3123860359f, 4.3122763634f, 4.3121666908f, 4.3120574951f, 4.3119492531f, 4.3118410110f,
4.3117337227f, 4.3116269112f, 4.3115200996f, 4.3114142418f, 4.3113088608f, 4.3112034798f, 4.3110990524f, 4.3109951019f, 4.3108911514f, 4.3107881546f,
4.3106851578f, 4.3105831146f, 4.3104810715f, 4.3103799820f, 4.3102788925f, 4.3101782799f, 4.3100786209f, 4.3099789619f, 4.3098797798f, 4.3097810745f,
4.3096828461f, 4.3095850945f, 4.3094873428f, 4.3093905449f, 4.3092937469f, 4.3091979027f, 4.3091020584f, 4.3090066910f, 4.3089118004f, 4.3088173866f,
4.3087234497f, 4.3086295128f, 4.3085360527f, 4.3084430695f, 4.3083505630f, 4.3082585335f, 4.3081669807f, 4.3080754280f, 4.3079848289f, 4.3078942299f,
4.3078041077f, 4.3077139854f, 4.3076248169f, 4.3075356483f, 4.3074469566f, 4.3073587418f, 4.3072705269f, 4.3071827888f, 4.3070955276f, 4.3070087433f,
4.3069224358f, 4.3068361282f, 4.3067502975f, 4.3066649437f, 4.3065795898f, 4.3064951897f, 4.3064107895f, 4.3063263893f, 4.3062429428f, 4.3061594963f,
4.3060760498f, 4.3059935570f, 4.3059110641f, 4.3058290482f, 4.3057475090f, 4.3056659698f, 4.3055849075f, 4.3055038452f, 4.3054237366f, 4.3053436279f,
4.3052635193f, 4.3051838875f, 4.3051047325f, 4.3050260544f, 4.3049473763f, 4.3048691750f, 4.3047909737f, 4.3047137260f, 4.3046360016f, 4.3045592308f,
4.3044824600f, 4.3044056892f, 4.3043298721f, 4.3042540550f, 4.3041782379f, 4.3041028976f, 4.3040280342f, 4.3039531708f, 4.3038787842f, 4.3038048744f,
4.3037309647f, 4.3036570549f, 4.3035840988f, 4.3035106659f, 4.3034381866f, 4.3033657074f, 4.3032932281f, 4.3032212257f, 4.3031497002f, 4.3030781746f,
4.3030071259f, 4.3029365540f, 4.3028655052f, 4.3027954102f, 4.3027253151f, 4.3026556969f, 4.3025860786f, 4.3025164604f, 4.3024473190f, 4.3023786545f,
4.3023099899f, 4.3022418022f, 4.3021736145f, 4.3021059036f, 4.3020381927f, 4.3019709587f, 4.3019042015f, 4.3018369675f, 4.3017706871f, 4.3017044067f,
4.3016381264f, 4.3015723228f, 4.3015065193f, 4.3014411926f, 4.3013758659f, 4.3013110161f, 4.3012461662f, 4.3011817932f, 4.3011174202f, 4.3010535240f,
4.3009896278f, 4.3009262085f, 4.3008627892f, 4.3007998466f, 4.3007369041f, 4.3006739616f, 4.3006114960f, 4.3005495071f, 4.3004875183f, 4.3004255295f,
4.3003640175f, 4.3003025055f, 4.3002414703f, 4.3001804352f, 4.3001194000f, 4.3000588417f, 4.2999987602f, 4.2999386787f, 4.2998785973f, 4.2998189926f,
4.2997593880f, 4.2996997833f, 4.2996406555f, 4.2995820045f, 4.2995233536f, 4.2994647026f, 4.2994065285f, 4.2993483543f, 4.2992901802f, 4.2992324829f,
4.2991752625f, 4.2991175652f, 4.2990603447f, 4.2990036011f, 4.2989468575f, 4.2988901138f, 4.2988338470f, 4.2987775803f, 4.2987213135f, 4.2986655235f,
4.2986102104f, 4.2985544205f, 4.2984991074f, 4.2984442711f, 4.2983894348f, 4.2983345985f, 4.2982797623f, 4.2982254028f, 4.2981710434f, 4.2981171608f,
4.2980632782f, 4.2980098724f, 4.2979559898f, 4.2979025841f, 4.2978496552f, 4.2977967262f, 4.2977437973f, 4.2976908684f, 4.2976384163f, 4.2975864410f,
4.2975339890f, 4.2974820137f, 4.2974300385f, 4.2973785400f, 4.2973270416f, 4.2972755432f, 4.2972245216f, 4.2971735001f, 4.2971224785f, 4.2970719337f,
4.2970213890f, 4.2969713211f, 4.2969207764f, 4.2968707085f, 4.2968211174f, 4.2967710495f, 4.2967214584f, 4.2966718674f, 4.2966227531f, 4.2965736389f,
4.2965245247f, 4.2964758873f, 4.2964272499f, 4.2963786125f, 4.2963304520f, 4.2962818146f, 4.2962341309f, 4.2961859703f, 4.2961382866f, 4.2960906029f,
4.2960429192f, 4.2959957123f, 4.2959485054f, 4.2959012985f, 4.2958545685f, 4.2958078384f, 4.2957611084f, 4.2957143784f, 4.2956681252f, 4.2956218719f,
4.2955756187f, 4.2955298424f, 4.2954840660f, 4.2954382896f, 4.2953929901f, 4.2953476906f, 4.2953023911f, 4.2952570915f, 4.2952122688f, 4.2951669693f,
4.2951226234f, 4.2950778008f, 4.2950334549f, 4.2949891090f, 4.2949447632f, 4.2949008942f, 4.2948565483f, 4.2948131561f, 4.2947692871f, 4.2947258949f,
4.2946820259f, 4.2946391106f, 4.2945957184f, 4.2945528030f, 4.2945098877f, 4.2944669724f, 4.2944240570f, 4.2943816185f, 4.2943391800f, 4.2942967415f,
4.2942547798f, 4.2942128181f, 4.2941708565f, 4.2941288948f, 4.2940869331f, 4.2940454483f, 4.2940039635f, 4.2939624786f, 4.2939214706f, 4.2938804626f,
4.2938394547f, 4.2937984467f, 4.2937574387f, 4.2937169075f, 4.2936763763f, 4.2936358452f, 4.2935957909f, 4.2935552597f, 4.2935152054f, 4.2934751511f,
4.2934355736f, 4.2933955193f, 4.2933559418f, 4.2933163643f, 4.2932767868f, 4.2932376862f, 4.2931985855f, 4.2931594849f, 4.2931203842f, 4.2930812836f,
4.2930426598f, 4.2930040359f, 4.2929654121f, 4.2929267883f, 4.2928886414f, 4.2928504944f, 4.2928123474f, 4.2927742004f, 4.2927360535f, 4.2926983833f,
4.2926607132f, 4.2926230431f, 4.2925853729f, 4.2925477028f, 4.2925105095f, 4.2924733162f, 4.2924361229f, 4.2923994064f, 4.2923622131f, 4.2923254967f,
4.2922887802f, 4.2922520638f, 4.2922153473f, 4.2921791077f, 4.2921428680f, 4.2921066284f, 4.2920703888f, 4.2920341492f, 4.2919983864f, 4.2919626236f,
4.2919268608f, 4.2918910980f, 4.2918553352f, 4.2918200493f, 4.2917842865f, 4.2917490005f, 4.2917141914f, 4.2916789055f, 4.2916436195f, 4.2916088104f,
4.2915740013f, 4.2915391922f, 4.2915048599f, 4.2914700508f, 4.2914357185f, 4.2914013863f, 4.2913670540f, 4.2913327217f, 4.2912983894f, 4.2912645340f,
4.2912306786f, 4.2911968231f, 4.2911629677f, 4.2911291122f, 4.2910957336f, 4.2910618782f, 4.2910284996f, 4.2909951210f, 4.2909622192f, 4.2909288406f,
4.2908959389f, 4.2908625603f, 4.2908296585f, 4.2907967567f, 4.2907643318f, 4.2907314301f, 4.2906990051f, 4.2906665802f, 4.2906341553f, 4.2906017303f,
4.2905693054f, 4.2905373573f, 4.2905049324f, 4.2904729843f, 4.2904410362f, 4.2904090881f, 4.2903776169f, 4.2903456688f, 4.2903141975f, 4.2902827263f,
4.2902512550f, 4.2902197838f, 4.2901883125f, 4.2901573181f, 4.2901258469f, 4.2900948524f, 4.2900638580f, 4.2900328636f, 4.2900018692f, 4.2899713516f,
4.2899408340f, 4.2899098396f, 4.2898793221f, 4.2898488045f, 4.2898187637f, 4.2897882462f, 4.2897577286f, 4.2897276878f, 4.2896976471f, 4.2896676064f,
4.2896375656f, 4.2896075249f, 4.2895779610f, 4.2895479202f, 4.2895183563f, 4.2894887924f, 4.2894592285f, 4.2894296646f, 4.2894005775f, 4.2893710136f,
4.2893419266f, 4.2893128395f, 4.2892837524f, 4.2892546654f, 4.2892255783f, 4.2891964912f, 4.2891678810f, 4.2891392708f, 4.2891101837f, 4.2890815735f,
4.2890529633f, 4.2890248299f, 4.2889962196f, 4.2889676094f, 4.2889394760f, 4.2889113426f, 4.2888832092f, 4.2888550758f, 4.2888269424f, 4.2887988091f,
4.2887711525f, 4.2887434959f, 4.2887153625f, 4.2886877060f, 4.2886600494f, 4.2886323929f, 4.2886052132f, 4.2885775566f, 4.2885503769f, 4.2885227203f,
4.2884955406f, 4.2884683609f, 4.2884411812f, 4.2884140015f, 4.2883872986f, 4.2883601189f, 4.2883334160f, 4.2883067131f, 4.2882795334f, 4.2882528305f,
4.2882266045f, 4.2881999016f, 4.2881731987f, 4.2881469727f, 4.2881202698f, 4.2880940437f, 4.2880678177f, 4.2880415916f, 4.2880153656f, 4.2879891396f,
4.2879633904f, 4.2879371643f, 4.2879114151f, 4.2878856659f, 4.2878594398f, 4.2878336906f, 4.2878084183f, 4.2877826691f, 4.2877569199f, 4.2877316475f,
4.2877058983f, 4.2876806259f, 4.2876553535f, 4.2876300812f, 4.2876048088f, 4.2875795364f, 4.2875542641f, 4.2875294685f, 4.2875041962f, 4.2874794006f,
4.2874541283f, 4.2874293327f, 4.2874045372f, 4.2873797417f, 4.2873554230f, 4.2873306274f, 4.2873058319f, 4.2872815132f, 4.2872571945f, 4.2872323990f,
4.2872080803f, 4.2871837616f, 4.2871594429f, 4.2871356010f, 4.2871112823f, 4.2870869637f, 4.2870631218f, 4.2870392799f, 4.2870149612f, 4.2869911194f,
4.2869672775f, 4.2869434357f, 4.2869200706f, 4.2868962288f, 4.2868723869f, 4.2868490219f, 4.2868256569f, 4.2868018150f, 4.2867784500f, 4.2867550850f,
4.2867317200f, 4.2867083549f, 4.2866854668f, 4.2866621017f, 4.2866387367f, 4.2866158485f, 4.2865929604f, 4.2865695953f, 4.2865467072f, 4.2865238190f,
4.2865009308f, 4.2864780426f, 4.2864556313f, 4.2864327431f, 4.2864103317f, 4.2863874435f, 4.2863650322f, 4.2863426208f, 4.2863202095f, 4.2862977982f,
4.2862753868f, 4.2862529755f, 4.2862305641f, 4.2862081528f, 4.2861862183f, 4.2861638069f, 4.2861418724f, 4.2861199379f, 4.2860980034f, 4.2860760689f,
4.2860541344f, 4.2860321999f, 4.2860102654f, 4.2859883308f, 4.2859668732f, 4.2859449387f, 4.2859234810f, 4.2859020233f, 4.2858805656f, 4.2858586311f,
4.2858371735f, 4.2858157158f, 4.2857947350f, 4.2857732773f, 4.2857518196f, 4.2857308388f, 4.2857093811f, 4.2856884003f, 4.2856674194f, 4.2856459618f,
4.2856249809f, 4.2856040001f, 4.2855830193f, 4.2855620384f, 4.2855415344f, 4.2855205536f, 4.2854995728f, 4.2854790688f, 4.2854585648f, 4.2854375839f
};

float cached_t_quantile(int v) {
    if (v < 1) {
        return z_lookup[0];
    }
    if (v < z_entries) {
        return z_lookup[v - 1];
    }
    // z approaches constant when v is high enough.
    // With default lookup table size the function is flat enough that we
    // can just return the last entry for all v bigger than it.
    return z_lookup[z_entries - 1];
}

static std::mutex IOMutex;

Logging::Logging(const char* file, int line, bool write_only) {
    m_file = std::string{file};
    m_line = line;
    m_write_only = write_only;
}

Logging::~Logging() {
    std::lock_guard<std::mutex> lock(IOMutex);
    if (!m_write_only) {
        std::cout << str() << std::flush;
    }
    if (!option<std::string>("log_file").empty()) {
        auto fp = std::fstream{};  
        fp.open(option<std::string>("log_file"), std::ios::app | std::ios::out);
        if (fp.is_open()) {
            fp << str();
            fp.close();
        }
    }
}

StandError::StandError(const char* file, int line) {
    m_file = std::string{file};
    m_line = line;
}

StandError::~StandError() {
    std::lock_guard<std::mutex> lock(IOMutex);
    std::cerr << str() << std::flush;
    if (!option<std::string>("log_file").empty()) {
        auto fp = std::fstream{};  
        fp.open(option<std::string>("log_file"), std::ios::app | std::ios::out);
        if (fp.is_open()) {
            fp << str();
            fp.close();
        }
    }
}

UCCIDebug::UCCIDebug(const char* file, int line) {
    m_file = std::string{file};
    m_line = line;
}

UCCIDebug::~UCCIDebug() {
    if (!option<bool>("debug_verbose")) {
        return;
    }

    std::lock_guard<std::mutex> lock(IOMutex);
    if (!option<std::string>("log_file").empty()) {
        auto fp = std::fstream{};  
        fp.open(option<std::string>("log_file"), std::ios::app | std::ios::out);
        if (fp.is_open()) {
            fp << str();
            fp.close();
        }
    } else {
        std::cout << str() << std::flush;
    }
}

std::string bool_to_str(bool v) {
    if (v) {
        return std::string{"true"};
    }
    return std::string{"false"};
}

void space_stream(std::ostream &out, const size_t times) {
    for (auto t = size_t{0}; t < times; ++t) {
        out << " ";
    }
}

void strip_stream(std::ostream &out, const size_t times) {
    for (auto t = size_t{0}; t < times; ++t) {
        out << std::endl;
    }
}

constexpr size_t CommandParser::MAX_BUFFER_SIZE;

CommandParser::CommandParser(std::string &input) {
    parse(std::forward<std::string>(input), MAX_BUFFER_SIZE);
}

CommandParser::CommandParser(std::string &input, const size_t max) {
    parse(std::forward<std::string>(input), std::min(max, MAX_BUFFER_SIZE));
}

CommandParser::CommandParser(int argc, char** argv) {
    auto out = std::ostringstream{};
    for (int i = 0; i < argc; ++i) {
        out << argv[i] << " ";
    }
    parse(std::forward<std::string>(out.str()), MAX_BUFFER_SIZE);
}

bool CommandParser::valid() const {
    return m_count != 0;
}

void CommandParser::parse(std::string &input, const size_t max) {
    m_count = 0;
    auto stream = std::istringstream{input};
    auto in = std::string{};
    while (stream >> in) {
        m_commands.emplace_back(std::make_shared<std::string>(in));
        m_count++;
        if (m_count >= max) break;
    }
}

void CommandParser::parse(std::string &&input, const size_t max) {
    m_count = 0;
    auto stream = std::istringstream{input};
    auto in = std::string{};
    while (stream >> in) {
        m_commands.emplace_back(std::make_shared<std::string>(in));
        m_count++;
        if (m_count >= max) break;
    }
}

size_t CommandParser::get_count() const {
    return m_count;
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::get_command(size_t id) const {
    if (!valid() || id >= m_count) {
        return nullptr;
    }
    return std::make_shared<Reuslt>(Reuslt(*m_commands[id], (int)id));
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::get_commands(size_t b) const {
    return get_slice(b, m_count);
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::get_slice(size_t b, size_t e) const {
     if (!valid() || b >= m_count || e > m_count || b >= e) {
         return nullptr;
     }

     auto out = std::ostringstream{};
     auto begin = std::next(std::begin(m_commands), b);
     auto end = std::next(std::begin(m_commands), e);
     auto stop = std::prev(end, 1);

     if (begin != end) {
         std::for_each(begin, stop, [&](auto in)
                                        {  out << *in << " "; });
     }

     out << **stop;
     return std::make_shared<Reuslt>(Reuslt(out.str(), -1));
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::find(const std::string input, int id) const {
    if (!valid()) {
        return nullptr;
    }

    if (id < 0) {
        for (auto i = size_t{0}; i < get_count(); ++i) {
            const auto res = get_command((size_t)i);
            if (res->str == input) {
                return res;
            }
        }
    } else {
        if (const auto res = get_command((size_t)id)) {
            return res->str == input ? res : nullptr;
        }
    }
    return nullptr;
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::find(const std::initializer_list<std::string> inputs, int id) const {
    for (const auto &in : inputs) {
        if (const auto res = find(in, id)) {
            return res;
        }
    }
    return nullptr;
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::find_next(const std::string input) const {
    const auto res = find(input);

    if (!res || res->idx+1 > (int)get_count()) {
        return nullptr;
    }
    return get_command(res->idx+1);
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::find_next(const std::initializer_list<std::string> inputs) const {
    for (const auto &in : inputs) {
        if (const auto res = find_next(in)) {
            return res;
        }
    }
    return nullptr;
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::remove_command(size_t id) {
    if (id > get_count()) {
        return nullptr;
    }

    const auto str = *m_commands[id];
    m_commands.erase(std::begin(m_commands)+id);
    m_count--;

    return std::make_shared<Reuslt>(Reuslt(str, -1));
}

std::shared_ptr<CommandParser::Reuslt> CommandParser::remove_slice(size_t b, size_t e) {
    if (b > get_count() || e > get_count() || b > e) {
        return nullptr;
    }
    if (b == e) {
        return remove_command(e);
    }
    auto out = get_slice(b, e);
    m_commands.erase(std::begin(m_commands)+b, std::begin(m_commands)+e);
    m_count -= (e-b);
    return out;
}

template<>
std::string CommandParser::Reuslt::get<std::string>() const {
    return str;
}

template<>
int CommandParser::Reuslt::get<int>() const {
    return std::stoi(str);
}

template<>
float CommandParser::Reuslt::get<float>() const{
    return std::stof(str);
}

template<>
char CommandParser::Reuslt::get<char>() const{
    return str[0];
}

template<>
const char* CommandParser::Reuslt::get<const char*>() const{
    return str.c_str();
}

bool Option::boundary_valid() const {
    option_handle();
    return !(m_max == 0 && m_min == 0);
}

template<>
Option Option::setoption<std::string>(std::string val, int /*max*/, int /*min*/) {
    return Option{type::String, val, 0, 0};
}

template<>
Option Option::setoption<const char *>(const char *val, int /*max*/, int /*min*/) {
    return Option{type::String, std::string{val}, 0, 0};
}

template<>
Option Option::setoption<bool>(bool val, int /*max*/, int /*min*/) {
    if (val) {
        return Option{type::Bool, "true", 0, 0};
    }
    return Option{type::Bool, "false", 0, 0};
}

template<>
Option Option::setoption<int>(int val, int max, int min) {
    auto op = Option{type::Integer, std::to_string(val), max, min};
    op.adjust<int>();
    return op;
}

template<>
Option Option::setoption<float>(float val, int max, int min) {
    auto op = Option{type::Float, std::to_string(val), max, min};
    op.adjust<float>();
    return op;
}

template<>
Option Option::setoption<char>(char val, int /*max*/, int /*min*/) {
    return Option{type::Char, std::string{val}, 0, 0};
}

#define OPTION_EXPASSION(T) \
template<>                  \
T Option::get<T>() const {  \
    return (T)*this;        \
}

OPTION_EXPASSION(std::string)
OPTION_EXPASSION(bool)
OPTION_EXPASSION(float)
OPTION_EXPASSION(int)
OPTION_EXPASSION(char)

template<>
const char* Option::get<const char*>() const {
    return m_value.c_str();
}

#undef OPTION_EXPASSION

template<>
void Option::set<std::string>(std::string value) {
    option_handle();
    m_value = value;
}

template<>
void Option::set<bool>(bool value) {
    option_handle();
    if (value) {
        m_value = std::string{"true"};
    } else {
        m_value = std::string{"false"};
    }
}

template<>
void Option::set<int>(int value) {
    option_handle();
    m_value = std::to_string(value);
    adjust<int>();
}

template<>
void Option::set<float>(float value) {
    option_handle();
    m_value = std::to_string(value);
    adjust<float>();
}

template<>
void Option::set<char>(char value) {
    option_handle();
    m_value = std::string{value};
}

void Option::option_handle() const {
    if (m_max < m_min) {
        auto out = std::ostringstream{};
        out << " Option Error :";
        out << " Max : " << m_max << " |";
        out << " Min : " << m_min << " |";
        out << " Minimal is bigger than maximal.";
        out << " It is not accepted.";
        throw std::runtime_error(out.str());
    }

    if (m_type == type::Invalid) {
        auto out = std::ostringstream{};
        out << " Option Error :";
        out << " Please initialize first.";
        throw std::runtime_error(out.str());
    }
};

Timer::Timer() {
    clock();
    record_count = 0;
}

void Timer::clock() {
    m_clock_time = std::chrono::steady_clock::now();
}

int Timer::get_duration_seconds() const {
    const auto end_time = std::chrono::steady_clock::now();
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - m_clock_time).count();
    return seconds;
}

int Timer::get_duration_milliseconds() const {
    const auto end_time = std::chrono::steady_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - m_clock_time).count();
    return milliseconds;
}

int Timer::get_duration_microseconds() const {
    const auto end_time = std::chrono::steady_clock::now();
    const auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - m_clock_time).count();
    return microseconds;
}


float Timer::get_duration() const {
    const auto seconds = get_duration_seconds();
    const auto milliseconds = get_duration_milliseconds();
    if (seconds == (milliseconds/1000)) {
        return static_cast<float>(milliseconds) / 1000.f;
    } else {
        return static_cast<float>(seconds);
    }
}

void Timer::record() {
    m_record.emplace_back(std::move(get_duration()));
    record_count++;
    assert(m_record.size() == record_count);
}

void Timer::release() {
    m_record.clear();
    record_count = 0;
}

float Timer::get_record_time(size_t id) const {
    if (record_count == 0) {
        return 0.f;
    }
    if (id > record_count) {
        id = record_count;
    } 
    else if (id <= 0) {
        id = 1;
    }
    return m_record[id-1];
}

int Timer::get_record_count() const {
    return record_count;
}

const std::string get_current_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d-%X", &tstruct);

    return buf;
}

const std::vector<float>& Utils::Timer::get_record() const {
    return m_record;
}

BitIterator::BitIterator(const size_t s) {
    if (s < 64) {
        m_size = s;
    } else {
        m_size = 64;
    }
    set(0ULL);
}

std::vector<bool> BitIterator::get() const {
    auto res = std::vector<bool>{};
    for (auto i = size_t{0}; i < m_size; ++i) {
        res.emplace_back(bit_signed(i));
    }
    return res;
}

void BitIterator::set(std::uint64_t cnt) {
    m_cnt = cnt;
}

bool BitIterator::bit_signed(size_t s) const {
    return (m_cnt >> (m_size - s - 1)) & 1ULL;
}

void BitIterator::next() {
    m_cnt++;
}

void BitIterator::back() {
    m_cnt--;
}

} // namespace Utils
