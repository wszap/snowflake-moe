#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_gen_synth.py — Snowflake MoE 训练用合成数据集生成器
==========================================================
用规则模板 + 精心挑选的语料库，离线合成三种风格鲜明的英文文本：
  1) bible_kjv.txt        KJV 圣经风格（创世/先知/诗篇/福音/书信）  ~5MB
  2) data_shk_comedy.txt  莎士比亚喜剧风格（误会/伪装/圆满结局）      ~2MB
  3) data_shk_history.txt 莎士比亚历史剧风格（王权/战争/政治阴谋）    ~2MB

特性：
  - 纯标准库，无外部依赖；确定性 seed 可复现
  - 只输出 ASCII 字符，与 TinyStories 字符集兼容
  - 每段 200–500 字符；按目标大小自动续写（已有文件则断点续跑）
  - 不占用 GPU

用法：
  python data_gen_synth.py --style bible    --out /path/bible_kjv.txt    --target-mb 5
  python data_gen_synth.py --style comedy   --out /path/data_shk_comedy.txt --target-mb 2
  python data_gen_synth.py --style history  --out /path/data_shk_history.txt --target-mb 2
  python data_gen_synth.py --all            --out /path                 --target-mb 5,2,2
"""

import argparse
import os
import random
import sys

# ----------------------------------------------------------------------------
# 语料库
# ----------------------------------------------------------------------------

# ---------------- 圣经 KJV 风格 ----------------
BIBLE_OPEN = [
    "And it came to pass in those days,",
    "And the LORD spake unto {n1}, saying,",
    "The word of the LORD came unto {n1}, saying,",
    "Behold, I will",
    "Hearken unto me, ye children of Israel,",
    "Blessed is the man that",
    "Verily, verily, I say unto you,",
    "In the beginning",
    "And God said,",
    "Then spake {n1} and said,",
    "And it came to pass, when {n1} heard these words,",
    "Thus saith the LORD, the God of Israel,",
    "And there was a famine in the land,",
    "And they journeyed from {p1} toward {p2},",
    "Now it came to pass after many days,",
    "And the angel of the LORD appeared unto {n1}, saying,",
    "Hear the word of the LORD, ye rulers of the people,",
    "The LORD is my {n2}; I shall not want.",
    "Sing unto the LORD a new song;",
    "And it was told the king, saying,",
]

BIBLE_MID = [
    "that walketh not in the counsel of the ungodly, nor standeth in the way of sinners, nor sitteth in the seat of the scornful.",
    "for the land is given into the hand of the wicked, and the righteous shall inherit the earth in due season.",
    "and the waters prevailed exceedingly upon the earth; and all the high hills that were under the whole heaven were covered.",
    "and they shall beat their swords into plowshares, and their spears into pruninghooks: nation shall not lift up sword against nation, neither shall they learn war any more.",
    "for thou shalt serve the LORD thy God with all thine heart, and with all thy soul, and with all thy might.",
    "and I will bring again the captivity of my people, and they shall build the waste cities, and dwell in them.",
    "for the LORD is good; his mercy is everlasting; and his truth endureth to all generations.",
    "and the shepherd shall smite the flock, and the sheep shall be scattered abroad; yet will I gather the remnant of them.",
    "and they took the young men of the city, and taught them the way of the LORD, that they might walk therein.",
    "and there arose a great storm upon the sea, and the ship was like to be broken; but the LORD rebuked the wind, and there was a great calm.",
    "and the people went forth, and gathered manna every morning, every man according to his eating; and when the sun waxed hot, it melted.",
    "for I know the thoughts that I think toward you, saith the LORD, thoughts of peace, and not of evil, to give you an expected end.",
    "and the king made a decree throughout all his kingdom, that every man should bear rule in his own house.",
    "and the prophet cried with a loud voice, saying, Turn ye, turn ye from your evil ways; for why will ye die?",
    "and they laid wait for him in the way, and sought to take him by subtlety; but the LORD delivered him out of their hand.",
    "and the widow's cruse of oil failed not, neither did the barrel of meal waste, according to the word of the LORD which he spake by {n1}.",
    "and the young man said, Father, I have sinned against heaven and before thee, and am no more worthy to be called thy son.",
    "and the multitude marvelled, and glorified God, saying, We never saw it on this fashion.",
    "and the nations shall hear of thy name, and shall tremble, and shall be moved at the report of thy judgments.",
    "and they sowed in tears, and reaped in joy; he that goeth forth weeping, bearing precious seed, shall doubtless come again rejoicing.",
]

BIBLE_END = [
    "And it was so.",
    "Thus saith the LORD.",
    "Praise ye the LORD.",
    "For the LORD is good; his mercy endureth for ever.",
    "Amen.",
    "And the people answered with one voice, and said, Amen, and praised the LORD.",
    "And he blessed them, and they went their way.",
    "And the word of the LORD was precious in those days; there was no open vision.",
    "And they rested from all their works, and the land had rest.",
    "And it came to pass, that as he sowed, the seed fell upon good ground, and brought forth fruit, some an hundredfold.",
    "And the land was filled with the knowledge of the LORD, as the waters cover the sea.",
    "And the king said, Let the word stand; and it was established among the people.",
]

BIBLE_NAMES = [
    "Abraham", "Isaac", "Jacob", "Joseph", "Moses", "Aaron", "Joshua", "Samuel",
    "David", "Solomon", "Elijah", "Elisha", "Isaiah", "Jeremiah", "Ezekiel",
    "Daniel", "Hosea", "Amos", "Jonah", "Micah", "Zechariah", "Malachi",
    "Peter", "John", "Paul", "James", "Timothy", "Barnabas", "Silas",
]

BIBLE_PLACES = [
    "Haran", "Canaan", "Egypt", "Bethel", "Jerusalem", "Zion", "Hebron",
    "Gilgal", "Jericho", "Damascus", "Babylon", "Nineveh", "Gilead", "Moab",
]

BIBLE_VIRTUES = [
    "strength", "rock", "shield", "refuge", "light", "salvation", "shepherd",
    "portion", "song", "fortress", "deliverer", "hope", "peace", "truth",
]

BIBLE_VERBS = [
    "seek", "walk", "trust", "praise", "keep", "love", "fear", "serve",
    "call upon", "rejoice in", "wait upon", "meditate on", "delight in",
    "obey", "follow after", "lift up", "magnify", "declare",
]

# ---------------- 莎士比亚喜剧风格 ----------------
COMEDY_SPEAKERS = [
    "Benedick", "Beatrice", "Viola", "Orsino", "Rosalind", "Celia",
    "Touchstone", "Petruchio", "Katherine", "Lysander", "Hermia",
    "Portia", "Bassanio", "Nerissa", "Sebastian", "Olivia", "Antonio",
    "Feste", "Sir Toby", "Maria", "Jaques", "Duke Senior", "Oliver",
]

COMEDY_OPEN = [
    "Nay, by my troth, I know not wherefore I should be merry.",
    "Forsooth, and it please your grace, the matter is not so simple as it seems.",
    "Methinks I hear a voice within the garden, and it speaks of love and foolishness.",
    "By my faith, thou art a strange fellow; come, tell me what hath chanced betwixt us.",
    "Prithee, good cousin, lay aside this melancholy and join the revels of the evening.",
    "What says the messenger? Hath the masque been prepared, and the dancers all in readiness?",
    "In faith, I would not be thy enemy for the world; yet thou provokest me past all patience.",
    "Ah, my sweet lady, the very wind that blows from the south whispers of thy beauty.",
    "Come, come, no more of this; thou shalt be my partner in this merry plot.",
    "I marvel much, that having such a wit, thou shouldst use it to torment the gentle sex.",
    "Now, by the roguery of Cupid, I swear I will have an answer to this riddle.",
    "Why, this is strange: the lady, who but yesterday disdained all wooers, now smiles upon every suitor.",
]

COMEDY_MID = [
    "and thus the false report was spread abroad, that the one had sworn love to the other, when neither knew a word of the matter.",
    "but the disguised youth, who was in truth the lady herself, spake so movingly that the duke's heart was wholly won.",
    "and so the two were married ere the morning broke, to the great amazement of all that looked on.",
    "yet the servant, mistaking the letter, delivered it to the wrong hand, and thereby set the whole household in a buzz.",
    "and the suitor, being beaten at his own game, was fain to confess that wit is no match for a woman's cunning.",
    "whereupon the companions fell a-laughing, and swore they had never seen so strange a comedy of errors.",
    "and the lady, perceiving the trick, turned it upon the jester, so that he was forced to beg her pardon on his knees.",
    "for the two lovers, being parted by storm and mischance, were at length brought together in the selfsame wood.",
    "and the old man, whose humour had been sour, was so mollified by the wedding feast that he gave his blessing to all.",
    "but the letter, which was meant to cause a quarrel, was read with such good will that it ended in an embrace.",
    "and thus by a merry confusion of identities, every soul was matched to its true love before the play was done.",
    "for the disguises, once stripped away, revealed nothing more than a pair of fools who had loved each other all along.",
]

COMEDY_END = [
    "And so the evening ended with music, and all the company danced till the candles burned low.",
    "Thus the mist was cleared, and every heart found its counterpart; may heaven send them joy.",
    "So ends the comedy of errors; let us all laugh, and forgive, and feast.",
    "And they were married, and the wedding bells rang, and the fools were the wisest of the company.",
    "Come, let us to the banquet; for jesting hath sharpened our appetites.",
    "And so, by the turning of fortune's wheel, all sorrows were resolved in one happy hour.",
    "So they departed together, hand in hand, and the echo of their laughter lingered in the air.",
    "And the duke declared a holiday, and the whole court made merry till the dawn.",
    "Thus the plots and counterplots dissolved in mirth, and love triumphed over every obstacle.",
]

COMEDY_ADJ = [
    "merry", "gentle", "foolish", "witty", "tender", "wayward", "honest",
    "bright-eyed", "rosy-cheeked", "quick-witted", "soft-spoken", "playful",
    "jealous", "absurd", "whimsical", "gallant", "devilish", "sunny",
]

# ---------------- 莎士比亚历史剧风格 ----------------
HISTORY_SPEAKERS = [
    "King Henry", "King Richard", "Duke of York", "Duke of Gloucester",
    "Earl of Warwick", "Earl of Salisbury", "Lord Talbot", "Queen Margaret",
    "Prince Edward", "Lord Clifford", "Archbishop of York", "Duke of Somerset",
    "Bishop of Winchester", "Lord Say", "John of Gaunt", "Duke of Norfolk",
]

HISTORY_OPEN = [
    "This sceptre, though it be but a hollow staff, bears the weight of a bleeding kingdom.",
    "The crown sits heavy on the brow that wears it; for kings are but men, and men are but dust.",
    "Hear me, ye lords of England: the time hath come to draw the sword and make an end.",
    "What news from the field? Hath the battle turned, or doth the enemy press upon our rear?",
    "The trumpets sound, the banners wave, and England calls her sons to arms once more.",
    "I have seen the faces of ten thousand dead, and yet the quarrel is not ended.",
    "Treason lurks in every hall; a whisper in the ear may undo a kingdom.",
    "The parliament is summoned; let the great men of the realm give account of their counsels.",
    "My lords, we stand upon the edge of civil war; let us speak plainly ere it be too late.",
    "The heir is young, the nobles are divided, and the raven croaks above the tower.",
    "By the soul of my father, I will not see this realm torn asunder while I draw breath.",
    "The winter of our discontent may yet give way to a glorious summer, if we are resolute.",
]

HISTORY_MID = [
    "and the two hosts met upon the plain, and the arrows darkened the sun, and the ground grew red with blood.",
    "but the king, being counselled by traitors, gave ear to flattery, and the realm was brought to ruin.",
    "and the gates of the city were thrown open, and the victor entered with torch and sword.",
    "yet the young prince, escaped from the slaughter, swore by the cross to reclaim his father's crown.",
    "and the lords, perceiving the division, plotted each against the other, and the council chamber grew hot with accusation.",
    "whereupon the duke, being accused of treason, demanded trial by combat, and the challenge was accepted before the court.",
    "and the queen, in her grief, walked among the ruins, and called down heaven's vengeance upon the house that had wronged her.",
    "for the succession, being disputed, set cousin against cousin, and brother against brother, till the very throne was shaken.",
    "and the ambassador returned with terms of peace, but the peace was a hollow one, and war resumed within the year.",
    "and the siege was laid, and the engines battered the walls, and the defenders held fast through famine and fire.",
    "but the conspirators, being betrayed by one of their own, were taken and brought before the king's justice.",
    "and the king, looking upon the map of his shrunken dominions, wept, and said that England was no more than a garden plundered.",
]

HISTORY_END = [
    "And so the day was lost, and the hopes of the house were buried with the fallen.",
    "Thus ends the reign; let the chronicler record it, and let posterity judge.",
    "And the kingdom, having passed through fire, was at length united under one sceptre again.",
    "So the trumpets sounded once more, but this time they sounded for the dead.",
    "And the new king was crowned with the old crown, and the people knelt, and the pageant was complete.",
    "Thus fortune, which had smiled, turned her wheel, and the proud were cast down.",
    "And the chronicle records that from that day the realm knew peace, and the plough replaced the spear.",
    "So perished the last of that house; and the land, weary of war, welcomed a new line of kings.",
]

HISTORY_NPC = [
    "the bishop", "the earl", "the herald", "the captain", "the widow of the slain",
    "the exiled lord", "the young squire", "the veteran soldier", "the chancellor",
]

# ----------------------------------------------------------------------------
# 生成器
# ----------------------------------------------------------------------------

def _pick(rng, seq):
    return rng.choice(seq)


def _fill(tpl, rng):
    return tpl.format(
        n1=_pick(rng, BIBLE_NAMES),
        n2=_pick(rng, BIBLE_VIRTUES),
        p1=_pick(rng, BIBLE_PLACES),
        p2=_pick(rng, BIBLE_PLACES),
    )


def gen_bible_para(rng):
    parts = []
    parts.append(_fill(_pick(rng, BIBLE_OPEN), rng))
    # 2-4 个中段，避免相邻重复
    mids = rng.sample(BIBLE_MID, rng.randint(2, 4))
    for m in mids:
        parts.append(_fill(m, rng))
    parts.append(_pick(rng, BIBLE_END))
    return " ".join(parts) + "\n"


def gen_comedy_para(rng):
    parts = []
    parts.append(_pick(rng, COMEDY_OPEN))
    mids = rng.sample(COMEDY_MID, rng.randint(2, 4))
    parts.extend(mids)
    # 加一句角色对白收尾，提升戏剧感
    adj = _pick(rng, COMEDY_ADJ)
    spk = _pick(rng, COMEDY_SPEAKERS)
    parts.append(
        rng.choice([
            f'Whereupon {spk} laughed, and cried, "A {adj} end to a {adj} plot!"',
            f'Then said {spk}, "By my troth, this is the {adj}est jest I ever heard."',
            f'And {spk}, being {adj}, swore never to part from the company.',
        ])
    )
    parts.append(_pick(rng, COMEDY_END))
    return " ".join(parts) + "\n"


def gen_history_para(rng):
    parts = []
    parts.append(_pick(rng, HISTORY_OPEN))
    mids = rng.sample(HISTORY_MID, rng.randint(2, 4))
    parts.extend(mids)
    npc = _pick(rng, HISTORY_NPC)
    parts.append(
        rng.choice([
            f'And {npc} spake, saying, "This day shall be remembered in the chronicles."',
            f'But {npc} counselled patience, and the court fell silent.',
            f'Then {npc} drew near, and delivered the tidings with a heavy heart.',
        ])
    )
    parts.append(_pick(rng, HISTORY_END))
    return " ".join(parts) + "\n"


GENERATORS = {
    "bible": gen_bible_para,
    "comedy": gen_comedy_para,
    "history": gen_history_para,
}


def ascii_only(s):
    return all(ord(c) < 128 for c in s)


def generate(style, out_path, target_bytes, seed=2026):
    rng = random.Random(seed)
    gen = GENERATORS[style]

    # 断点续跑：统计已有字符数
    existing = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="ascii", errors="ignore") as f:
            existing = len(f.read())

    mode = "a" if existing > 0 else "w"
    n_paras = 0
    with open(out_path, mode, encoding="ascii") as f:
        while existing < target_bytes:
            para = gen(rng)
            if not ascii_only(para):
                raise RuntimeError("non-ASCII char generated! bug.")
            f.write(para)
            existing += len(para)
            n_paras += 1
            if n_paras % 500 == 0:
                print(f"  [{style}] {existing/1e6:.2f} MB / {target_bytes/1e6:.2f} MB, paras={n_paras}")

    size = os.path.getsize(out_path)
    print(f"[done] {style}: {size/1e6:.2f} MB, {n_paras} paras, ascii_only=True")
    return size


def main():
    ap = argparse.ArgumentParser(description="Snowflake MoE synthetic dataset generator")
    ap.add_argument("--style", choices=["bible", "comedy", "history", "all"])
    ap.add_argument("--out", required=True, help="output file (or dir with --all)")
    ap.add_argument("--target-mb", default="5,2,2", help="target sizes in MB for bible,comedy,history")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    targets = [float(x) for x in args.target_mb.split(",")]

    if args.style == "all":
        os.makedirs(args.out, exist_ok=True)
        plan = [
            ("bible", os.path.join(args.out, "bible_kjv.txt"), targets[0]),
            ("comedy", os.path.join(args.out, "data_shk_comedy.txt"), targets[1]),
            ("history", os.path.join(args.out, "data_shk_history.txt"), targets[2]),
        ]
    else:
        plan = [(args.style, args.out, targets[0])]

    for style, path, mb in plan:
        print(f"== generating {style} -> {path} ({mb} MB) ==")
        generate(style, path, int(mb * 1_000_000), seed=args.seed)

    print("ALL DONE")


if __name__ == "__main__":
    main()
