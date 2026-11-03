// 모바일 메뉴 열고 닫기
const menuBtn = document.querySelector('.btn_menu');
const gnb = document.getElementById('gnb');

function setMenu(open) {
  menuBtn.setAttribute('aria-expanded', open);
  menuBtn.querySelector('.blind').textContent = open ? '메뉴 닫기' : '메뉴 열기';
  gnb.classList.toggle('is-open', open);
}

menuBtn.addEventListener('click', () => setMenu(menuBtn.getAttribute('aria-expanded') !== 'true'));
// 메뉴에서 섹션을 고르면 메뉴를 닫는다
gnb.addEventListener('click', (e) => { if (e.target.closest('a')) setMenu(false); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setMenu(false); });

// PDF 내려받기 = 인쇄 창 (print 스타일시트로 A4에 맞춤)
// resume: 소개 + 프로젝트 요약 + 경력·학력·자격증, portfolio: 소개 + 프로젝트 상세
const PDF_TITLE = { resume: '김지연_이력서', portfolio: '김지연_포트폴리오' };
const pageTitle = document.title;

document.querySelectorAll('[data-print]').forEach((btn) => {
  btn.addEventListener('click', () => {
    setMenu(false);
    document.documentElement.dataset.print = btn.dataset.print;
    document.title = PDF_TITLE[btn.dataset.print];  // 저장 파일 이름이 된다
    window.print();
  });
});
window.addEventListener('afterprint', () => {
  delete document.documentElement.dataset.print;
  document.title = pageTitle;
});

// 프로필 사진 크게 보기
document.querySelectorAll('[data-dialog-open]').forEach((btn) => {
  const dialog = document.getElementById(btn.dataset.dialogOpen);
  btn.addEventListener('click', () => dialog.showModal());
  dialog.addEventListener('click', (e) => {
    // 바깥(배경)이나 닫기 버튼을 누르면 닫는다
    if (e.target === dialog || e.target.closest('[data-dialog-close]')) dialog.close();
  });
});

// ?print=resume / ?print=portfolio 로 열면 해당 PDF 구성으로 미리 맞춰 둔다 (링크로 바로 공유할 때)
const printParam = new URLSearchParams(location.search).get('print');
if (PDF_TITLE[printParam]) {
  document.documentElement.dataset.print = printParam;
  document.title = PDF_TITLE[printParam];
}
