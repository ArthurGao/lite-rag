"""`uv run python eval/make_xlang_hard.py .` writes eval/corpus-xlang-hard/{en,zh} and
eval/golden-xlang-hard.jsonl (S24): near-duplicate documents (tiers / regions / versions of one
policy differing only in numbers or conditions); every question names its variant, and its
evidence sentence exists in that document only."""

import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
EN, ZH = root / "eval/corpus-xlang-hard/en", root / "eval/corpus-xlang-hard/zh"
EN.mkdir(parents=True, exist_ok=True)
ZH.mkdir(parents=True, exist_ok=True)
docs, Q = {}, []


def doc(lang, id_, title, sections):
    text = f"# {title}\n\n" + "\n\n".join(f"## {h}\n\n{b}" for h, b in sections) + "\n"
    (EN if lang == "en" else ZH).joinpath(f"{id_}.md").write_text(text)
    docs[id_] = text


def q(id_, question, d, evidence, kind, lang, doc_lang):
    assert evidence in docs[d], (id_, evidence)
    others = [k for k, t in docs.items() if k != d and evidence in t]
    assert not others, (id_, others)
    Q.append(
        dict(
            id=id_,
            question=question,
            doc=d,
            evidence=evidence,
            kind=kind,
            lang=lang,
            doc_lang=doc_lang,
        )
    )


# --- English: Harbourline Cloud SLA tiers (near duplicates) ---
for tier, avail, credit, resp, window in [
    ("Standard", "99.5", "10", "eight business hours", "Sunday 02:00 to 06:00"),
    ("Premium", "99.9", "25", "two hours", "Sunday 03:00 to 05:00"),
    (
        "Enterprise",
        "99.95",
        "50",
        "thirty minutes",
        "the first Tuesday of each month, 01:00 to 02:00",
    ),
]:
    doc(
        "en",
        f"sla-{tier.lower()}",
        f"Harbourline Cloud {tier} service level agreement (fictional example)",
        [
            (
                "Availability",
                f"The {tier} tier commits to {avail}% monthly availability of the control plane.",
            ),
            (
                "Service credits",
                f"If {tier} availability falls below the commitment, the customer receives "
                f"a credit of {credit}% of that month's fee.",
            ),
            (
                "Support",
                f"A {tier} severity-one ticket receives a first human response within {resp}.",
            ),
            (
                "Maintenance",
                f"Planned maintenance for {tier} customers takes place on {window} (UTC).",
            ),
        ],
    )
# --- English: backup policy versions ---
for year, keep, copies, test in [
    ("2025", "35 days", "two regions", "every quarter"),
    ("2026", "90 days", "three regions", "every month"),
]:
    doc(
        "en",
        f"backup-{year}",
        f"Harbourline backup policy, {year} edition (fictional example)",
        [
            ("Retention", f"Under the {year} edition, daily snapshots are kept for {keep}."),
            ("Copies", f"The {year} edition stores each snapshot in {copies}."),
            ("Restore tests", f"Restore drills under the {year} edition run {test}."),
        ],
    )
# --- English: incident severities ---
for sev, page, update, review in [
    ("1", "within 5 minutes", "every 30 minutes", "within 5 business days"),
    ("2", "within 30 minutes", "every 2 hours", "within 15 business days"),
]:
    doc(
        "en",
        f"incident-sev{sev}",
        f"Harbourline incident handling, severity {sev} (fictional example)",
        [
            ("Paging", f"A severity-{sev} incident pages the on-call engineer {page}."),
            (
                "Status updates",
                f"During a severity-{sev} incident the status page is updated {update}.",
            ),
            ("Review", f"A written review of a severity-{sev} incident is due {review}."),
        ],
    )
doc(
    "en",
    "access-contractors",
    "Harbourline access for contractors (fictional example)",
    [
        (
            "Accounts",
            "Contractor accounts expire automatically after 60 days unless a manager renews them.",
        ),
        (
            "Production",
            "Contractors never receive standing access to production; each session needs a ticket.",
        ),
        (
            "Devices",
            "Contractors must connect from a managed virtual desktop, not their own laptops.",
        ),
    ],
)

# --- Chinese: 青柳 data-centre operations by region (near duplicates) ---
for region, temp, patrol, diesel in [
    ("华东", "二十二度", "每两小时", "七十二小时"),
    ("华北", "二十度", "每四小时", "四十八小时"),
    ("华南", "二十四度", "每小时", "九十六小时"),
]:
    doc(
        "zh",
        f"ops-{ {'华东': 'east', '华北': 'north', '华南': 'south'}[region] }",
        f"青柳{region}机房运维规范（虚构示例）",
        [
            ("温度", f"{region}机房冷通道的目标温度为{temp}。"),
            ("巡检", f"{region}机房的值班人员{patrol}巡检一次。"),
            ("备用电源", f"{region}机房的柴油储备须能支撑{diesel}满负荷运行。"),
        ],
    )
# --- Chinese: procurement tiers ---
for name, limit, quotes, approver in [
    ("小额", "五万元以下", "一家供应商报价即可", "部门经理"),
    ("大额", "五万元及以上", "至少三家供应商报价", "财务总监和总经理共同"),
]:
    doc(
        "zh",
        f"purchase-{ {'小额': 'small', '大额': 'large'}[name] }",
        f"青柳{name}采购流程（虚构示例）",
        [
            ("适用范围", f"{name}采购指单笔金额{limit}的采购。"),
            ("报价", f"{name}采购需要{quotes}。"),
            ("审批", f"{name}采购由{approver}审批。"),
        ],
    )
# --- Chinese: leave for probation vs regular staff ---
for who, annual, sick, notice in [
    ("试用期员工", "不享受年假", "每月一天带薪病假", "提前三天"),
    ("正式员工", "每年十五天年假", "每年十二天带薪病假", "提前两周"),
]:
    doc(
        "zh",
        f"leave-{ {'试用期员工': 'probation', '正式员工': 'regular'}[who] }",
        f"青柳{who}休假制度（虚构示例）",
        [
            ("年假", f"{who}{annual}。"),
            ("病假", f"{who}享有{sick}。"),
            ("申请", f"{who}请假需要{notice}在系统中提交申请。"),
        ],
    )


# questions: (id, question, doc, evidence-in-doc, kind, qlang, dlang)
def E(d, needle):
    """The line of document `d` that contains `needle`: a question's evidence."""
    return next(line for line in docs[d].splitlines() if needle in line)


cross_zh_en = [
    ("h01", "高级版（Premium）的控制面每月可用性承诺是多少？", "sla-premium", "99.9%"),
    (
        "h02",
        "企业版如果没达到可用性承诺，能拿到多少比例的服务抵扣？",
        "sla-enterprise",
        "credit of 50%",
    ),
    ("h03", "标准版一级工单多久会有人第一次回复？", "sla-standard", "eight business hours"),
    ("h04", "企业版客户的计划维护在什么时间进行？", "sla-enterprise", "first Tuesday"),
    ("h05", "2026 版备份制度里每日快照保留多久？", "backup-2026", "90 days"),
    ("h06", "2025 版备份制度的恢复演练多久做一次？", "backup-2025", "every quarter"),
    ("h07", "二级事故期间状态页多久更新一次？", "incident-sev2", "every 2 hours"),
    ("h08", "一级事故的书面复盘要在多久之内交？", "incident-sev1", "within 5 business days"),
    ("h09", "外包人员的账号多少天后自动失效？", "access-contractors", "60 days"),
    ("h10", "外包人员可以用自己的笔记本连接吗？", "access-contractors", "virtual desktop"),
    ("h11", "标准版的计划维护窗口是哪天几点？", "sla-standard", "Sunday 02:00"),
    ("h12", "2026 版备份把每个快照存在几个区域？", "backup-2026", "three regions"),
]
cross_en_zh = [
    (
        "h13",
        "What is the cold-aisle target temperature in the North China data centre?",
        "ops-north",
        "二十度",
    ),
    (
        "h14",
        "How often do on-duty staff patrol the South China data centre?",
        "ops-south",
        "每小时巡检",
    ),
    (
        "h15",
        "How many hours of full load must the diesel reserve cover in the East China data centre?",
        "ops-east",
        "七十二小时",
    ),
    ("h16", "How many supplier quotes does a large purchase need?", "purchase-large", "至少三家"),
    ("h17", "Who approves a small purchase?", "purchase-small", "部门经理"),
    ("h18", "Do employees on probation get annual leave?", "leave-probation", "不享受年假"),
    (
        "h19",
        "How much paid sick leave do regular employees get per year?",
        "leave-regular",
        "十二天带薪病假",
    ),
    (
        "h20",
        "How far ahead must a regular employee submit a leave request?",
        "leave-regular",
        "提前两周",
    ),
    ("h21", "What amount counts as a small purchase?", "purchase-small", "五万元以下"),
    ("h22", "How many hours must the North China diesel reserve last?", "ops-north", "四十八小时"),
    (
        "h23",
        "What is the target cold-aisle temperature in the East China data centre?",
        "ops-east",
        "二十二度",
    ),
    (
        "h24",
        "How far ahead must someone on probation request leave?",
        "leave-probation",
        "提前三天",
    ),
]
same = [
    (
        "h25",
        "Which tier gets a two-hour first response on severity-one tickets?",
        "sla-premium",
        "two hours",
        "en",
    ),
    (
        "h26",
        "What credit does the Standard tier pay when availability is missed?",
        "sla-standard",
        "credit of 10%",
        "en",
    ),
    (
        "h27",
        "How quickly is the on-call engineer paged for a severity-1 incident?",
        "incident-sev1",
        "within 5 minutes",
        "en",
    ),
    (
        "h28",
        "Under the 2025 backup edition, how long are daily snapshots kept?",
        "backup-2025",
        "35 days",
        "en",
    ),
    (
        "h29",
        "Can contractors have standing production access?",
        "access-contractors",
        "never receive standing access",
        "en",
    ),
    ("h30", "What availability does the Standard tier commit to?", "sla-standard", "99.5%", "en"),
    ("h31", "华南机房冷通道目标温度是多少？", "ops-south", "二十四度", "zh"),
    ("h32", "华北机房多久巡检一次？", "ops-north", "每四小时", "zh"),
    ("h33", "大额采购由谁审批？", "purchase-large", "财务总监和总经理", "zh"),
    ("h34", "试用期员工每月有几天带薪病假？", "leave-probation", "每月一天", "zh"),
    ("h35", "正式员工每年有多少天年假？", "leave-regular", "十五天年假", "zh"),
    ("h36", "华南机房柴油储备要支撑多久？", "ops-south", "九十六小时", "zh"),
]
for id_, question, d, needle in cross_zh_en:
    q(id_, question, d, E(d, needle), "cross", "zh", "en")
for id_, question, d, needle in cross_en_zh:
    q(id_, question, d, E(d, needle), "cross", "en", "zh")
for id_, question, d, needle, lang in same:
    q(id_, question, d, E(d, needle), "same", lang, lang)
(root / "eval/golden-xlang-hard.jsonl").write_text(
    "\n".join(json.dumps(x, ensure_ascii=False) for x in Q) + "\n"
)
print(len(docs), "docs,", len(Q), "questions")
