# المعمارية التقنية لمشروع ألفا (Project Alpha Architecture)

## 1. الواجهة الأمامية (Frontend)
- الإطار الأساسي: Next.js 15 مع React 19 وتطبيق TypeScript.
- إدارة الحالة والتصميم: Tailwind CSS مع Shadcn UI.
- الاتصال بالخادم: TanStack Query (React Query) للاتصالات المتزامنة والتخزين المؤقت.

## 2. الواجهة الخلفية والخدمات المصغرة (Backend Services)
- لغة الواجهة الخلفية الأساسية: Go (Golang) للخدمات عالية الأداء، وبايثون (Python FastAPI) لخدمات معالجة البيانات والذكاء الاصطناعي.
- بوابات الدفع (Payment Gateway): نستخدم Stripe و Apple Pay مع تفعيل بروتوكول 3D Secure 2.0.
- قاعدة البيانات الرئيسية: PostgreSQL مستضافة عبر Supabase مع قفل التكرار ومؤشرات pgvector للبحث الدلالي.
- التخزين المؤقت وقوائم الانتظار: Redis 7 لإدارة جلسات المستخدمين وقوائم انتظار مهام المعالجة الخلفية.

## 3. الأمان والنشر (DevOps & Security)
- إدارة الحاويات: Docker Multi-stage builds مع تشغيل الحاويات على Kubernetes (EKS).
- إدارة الأسرار: HashiCorp Vault؛ يُحظر تماماً وضع أي مفاتيح API داخل ملفات الكود أو مستودع Git.
