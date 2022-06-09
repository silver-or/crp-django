from django.db import models


class User(models.Model):  # 결정론적 알고리즘 (웹 장고) 에서 모델은 데이터베이스의 스키마
    use_in_migrations = True
    username = models.CharField(primary_key=True, max_length=10)
    password = models.CharField(max_length=10)
    name = models.TextField  # 문자열로 처리
    email = models.TextField
    regDate = models.DateField


    class Meta:  # 내부에 부모 클래스
        db_table = "users"

    def __str__(self):  # DB와 연결하는 도메인 객체 → __init__ 없음  # 자바의 toString
        return f'{self.pk}{self.username}'
