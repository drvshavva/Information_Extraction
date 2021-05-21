class Regex:
    VERB = r"(\b[a-z]{1,20}m[ıiuü][sş][a-z]*\b|\b[a-z]{1,20}[ae]c[ae][kgğ][a-z]*\b|\b[a-z]{1,20}m[ae]l[iı][a-z]*\b|\b[a-z]{1,20}yor[a-z]*\b|\b[a-z]{1,20}[dt][iı][nmgk\s][iı]\b)"
    PLACE = r"(\b[a-z]{1,20}[dt][ea][\sn]\b)"  # bulunma/ ayrılma/ çıkma hal eki
    WHEN = r"(\b[a-z]{1,20}[dt][iı][ğg][ıi]nd[ae][a-z]*\b|\b[a-z]{1,20}[dt][iı][ğg][ıi]\szaman*\b|\b[a-z]{1,20}yorken*\b|\b[a-z]{1,20}[ıi]nc[ae]*\b)"
    WHY = r"(\b[a-z]{1,20}[dt][iı][gğ][iı]\sicin*\b|\b[a-z]{1,20}[dt][ea]n\sdolayi*\b|\b[a-z]{1,20}[dt][ea]n\soturu*\b|\b[a-z]{1,20}[dt][ıi][gğ][iı]\sicin*\b|\b[a-z]{1,20}\sdiye*\b|\b[a-z]{1,20}[dt][ıi][ğg][iı]nd[ae]n*\b|\b[a-z]{1,20}\sicin*\b)"


class Default:
    ADJ = ["küçük", "büyük", "mutlu", "kirli", "uzun", "kısa", "temiz", "mutsuz", "yüksek", "alçak"]
    WHO = ["anne", "baba", "kahraman", "kardeş", "yiğit", "o", "bu", "onlar", "şu", "ben", "sen", "biz", "siz", "oğlan",
           "kız", "kadın", "erkek", "bay", "bayan"]
    PLACE = ["ev", "otel", "okul", "restoran", "market", "mekan", "cafe", "iş"]
    WHAT = ["uyur", "düşer", "bayılır", "oturur", "olur", "sıkılır"]


class Tables:
    ADJ = "sıfatlar"
    WHO = "kim"
    WHEN = "ne_zaman"
    WHAT = "ne_olur"
    VERBS = "fiiler"
    PLACE = "yer"
    WHY = "neden"

