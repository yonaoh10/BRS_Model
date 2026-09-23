"""Canned Hebrew banking dialogs used by the mock ASR engine.

These dialogs are also the source of the evaluation golden set
(`scripts/build_golden_set.py`), so each one deliberately seeds identifiers of
a different KIND:

  - dialog 0: a checksum-valid Israeli ID and a phone number  -> caught by SHAPE
  - dialog 1: a card's last four, a date of birth spoken in words, and a street
              address                                         -> shapeless
  - dialog 2: an account number and a mother's name given as a standalone
              answer to a verification question               -> shapeless

The shapeless ones have no recognisable form at all: the only signal that they
are identifiers is that a banker asked for them. They exist here so the
evaluation harness measures that path, not only the regex path.

All content is synthetic. No real customer data is in this file.
"""

from __future__ import annotations

# Each dialog: list of (speaker, text) in conversation order.
DIALOGS: list[list[tuple[str, str]]] = [
    # Dialog 0: account inquiry, includes PII (fake valid ID 123456782, phone).
    [
        ("banker", "שלום, הגעת לבנק, מדבר יועץ מהמוקד. במה אפשר לעזור?"),
        ("customer", "שלום, אני רוצה לבדוק למה ירדה לי עמלה חריגה בחשבון."),
        ("banker", "בשמחה אבדוק. לפני כן אני צריך לזהות אותך, מה מספר תעודת הזהות שלך?"),
        ("customer", "המספר הוא 123456782 והטלפון שלי הוא 052-1234567."),
        ("banker", "תודה, הזיהוי הושלם. אני רואה את החשבון, תן לי רגע לבדוק את החיוב."),
        ("customer", "בסדר, אני מחכה. זה חיוב של שלושים שקלים מלפני שבוע."),
        ("banker", "מצאתי. מדובר בעמלת פעולה בסניף. אפשר לעבור למסלול עמלות חודשי קבוע שיחסוך לך את זה."),
        ("customer", "כמה עולה המסלול הזה בחודש?"),
        ("banker", "המסלול עולה עשרה שקלים בחודש וכולל את כל פעולות הערוץ הישיר. אעביר אותך למסלול כבר היום אם תרצה."),
        ("customer", "כן, זה נשמע משתלם. תעביר אותי בבקשה."),
        ("banker", "בוצע. סיכום: עברת למסלול החודשי, החיוב הראשון בתחילת החודש הבא. יש עוד משהו שאוכל לעזור בו?"),
        ("customer", "לא, זה הכול. תודה רבה על העזרה."),
        ("banker", "בכיף, שיהיה לך יום נעים ותודה שפנית אלינו."),
    ],
    # Dialog 1: loan inquiry, banker asks clarifying questions.
    [
        ("banker", "בוקר טוב, מדבר בנקאי מצוות ההלוואות. איך אפשר לסייע?"),
        ("customer", "בוקר טוב. אני שוקל לקחת הלוואה לשיפוץ הדירה ורציתי להבין מה התנאים."),
        ("banker", "אשמח להסביר. קודם אזהה אותך בבקשה, מה ארבע הספרות האחרונות של הכרטיס ותאריך הלידה?"),
        ("customer", "הספרות הן 4580 ונולדתי בחמישי למרץ שמונים ושתיים."),
        ("banker", "ולאימות הכתובת, מה כתובת המגורים שלך?"),
        ("customer", "רחוב הרצל 15 בחיפה."),
        ("banker", "מעולה, הזיהוי הושלם. כמה כסף אתה צריך לשיפוץ ולאיזו תקופה נוח לך להחזיר?"),
        ("customer", "בערך שמונים אלף שקל, ואני חושב על החזר של חמש שנים."),
        ("banker", "הבנתי. במסלול הזה הריבית היום היא פריים פלוס אחוז וחצי, והיא יכולה להשתנות. יש גם עמלת פתיחת תיק של מאתיים שקל."),
        ("customer", "מה יהיה ההחזר החודשי בערך?"),
        ("banker", "ההחזר החודשי המשוער הוא כאלף וחמש מאות שקל. חשוב לי לציין שזו הערכה, והסכום הסופי ייקבע באישור הבקשה."),
        ("customer", "בסדר גמור. מה השלב הבא?"),
        ("banker", "אני פותח עבורך בקשה עכשיו, והתשובה תגיע עד יום עסקים אחד. אחזור אליך מחר עם עדכון. עוד שאלות?"),
        ("customer", "לא, תודה, מחכה לתשובה."),
        ("banker", "מצוין, נדבר מחר. יום טוב ושיהיה בהצלחה עם השיפוץ."),
    ],
    # Dialog 2: frustrated customer, card blocked - empathy exercised.
    [
        ("banker", "שלום, מוקד שירות הלקוחות. במה אפשר לעזור?"),
        ("customer", "הכרטיס שלי נחסם באמצע קנייה בסופר וזה ממש מביך. אני צריך פתרון עכשיו."),
        ("banker", "אני שומע שזה מאוד לא נעים, בוא נפתור את זה יחד עכשיו. רק אזהה אותך קודם, מה מספר החשבון שלך?"),
        ("customer", "מספר החשבון הוא 765432 בסניף הראשי."),
        ("banker", "תודה, ומה שם האם לאימות נוסף?"),
        ("customer", "רות."),
        ("banker", "הזיהוי הושלם, תודה על הסבלנות. אני בודק את הכרטיס עכשיו."),
        ("customer", "רק שתדע, אני לקוח כבר עשר שנים וזו פעם ראשונה שקורה לי דבר כזה."),
        ("banker", "אני מבין לגמרי את התסכול. מצאתי: הכרטיס נחסם אוטומטית בגלל ניסיון חיוב חריג בחו\"ל. זו הגנה על החשבון שלך."),
        ("customer", "לא הייתי בחו\"ל בכלל. מה עושים עכשיו?"),
        ("banker", "אני מבטל את החסימה ומזמין לך כרטיס חדש עם מספר חדש ליתר ביטחון. הוא יגיע עד שלושה ימי עסקים, ובינתיים הכרטיס הנוכחי ישוחרר לשימוש בארץ."),
        ("customer", "אוקיי, זה מרגיע. תודה על הטיפול המהיר."),
        ("banker", "בשמחה. לסיכום: החסימה הוסרה, כרטיס חדש בדרך אליך עד שלושה ימי עסקים. אם יהיה חיוב חשוד נוסף נתקשר אליך מיד. ערב טוב."),
    ],
]


def dialog_for_call(call_id: str) -> list[tuple[str, str]]:
    """Pick a dialog fixture deterministically per call_id."""
    index = sum(ord(c) for c in call_id) % len(DIALOGS)
    return DIALOGS[index]
