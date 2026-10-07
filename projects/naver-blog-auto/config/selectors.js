/**
 * 네이버 DOM 셀렉터 모음.
 * 네이버가 마크업을 바꾸면 발행/수집이 깨진다. 그때 고칠 곳은 여기 한 곳뿐이다.
 * 각 항목은 "후보 배열"이며 위에서부터 순서대로 시도한다.
 * (클래스명 뒤 해시는 네이버가 수시로 바꾸므로 접두사 매칭 ^= 을 함께 둔다)
 */
export const S = {
  login: {
    url: 'https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com',
    // 로그인 성공 판정에 쓰는 흔적들
    loggedInMarks: [
      '#account .MyView-module__link_login___HpHMW',
      '.MyView-module__my_area___Jh4vN',
      'a[href*="logout"]',
      '#gnb_logout_button',
    ],
    // 2차 인증/새 기기 등록 화면
    deviceRegister: ['#new\\.save', '#new\\.dontsave', '.btn_cancel'],
  },

  /**
   * 검색 결과는 클래스명에 해시가 붙는 sds-comps 디자인 시스템으로 바뀌어
   * (예: fender-ui_228e3bd1, KeP5TcBH886S8vXG) 클래스 셀렉터가 며칠도 못 간다.
   * 그래서 수집은 클래스가 아니라 **href 패턴과 문서 구조**로 뽑는다 (src/collect.js).
   * 여기서는 URL 빌더와, 그나마 의미가 담긴 클래스 힌트만 관리한다.
   */
  newsSearch: {
    url: (q, sort = 1) =>
      `https://search.naver.com/search.naver?where=news&query=${encodeURIComponent(q)}&sort=${sort}`,
    // 기사 링크가 아닌 것들 (도움말·로그인·Keep·언론사 안내 등)
    excludeHref: /search\.naver\.com|help\.naver\.com|nid\.naver\.com|keep\.naver\.com|news\.naver\.com\/main\/static|\/policy|^javascript:/,
    minTitleLen: 15,
  },

  blogSearch: {
    url: (q) =>
      `https://search.naver.com/search.naver?ssc=tab.blog.all&sm=tab_jum&query=${encodeURIComponent(q)}`,
    // 블로그 글 주소는 blog.naver.com/<아이디>/<글번호> 형태라 가장 믿을 만하다
    postHref: /blog\.naver\.com\/([^/?#]+)\/(\d{6,})/,
    authorHint: '.sds-comps-profile-info-title',
    minTitleLen: 8,
  },

  blogPost: {
    frame: '#mainFrame',
    body: ['.se-main-container', '#postViewArea', '.post-view', '.se_component_wrap'],
    title: ['.se-title-text', '.pcol1', '.htitle'],
  },

  imageSearch: {
    url: (q) =>
      `https://search.naver.com/search.naver?where=image&query=${encodeURIComponent(q)}`,
    thumb: ['.image_tile img', '.tile_item img', 'img.thumb', '.photo_bx img'],
  },

  editor: {
    // blog.naver.com/<id>?Redirect=Write 는 이제 블로그 홈으로 떨어진다.
    // GoBlogWrite.naver 가 로그인된 계정의 에디터로 알아서 보내주므로 이쪽이 안전하다.
    writeUrl: () => 'https://blog.naver.com/GoBlogWrite.naver',
    frame: '#mainFrame',
    // 진입 시 뜨는 방해 요소들 (있으면 닫는다)
    dismiss: [
      '.se-popup-button-cancel',              // "작성 중인 글" 불러오기 취소
      '.se-help-panel-close-button',          // 도움말 패널
      'button.se-popup-close-button',
      '.btn_close',
    ],
    title: ['.se-documentTitle .se-text-paragraph', '.se-section-documentTitle span.se-placeholder'],
    body: ['.se-component.se-text .se-text-paragraph', '.se-main-container .se-text-paragraph'],
    toolbar: {
      quote: ['button[data-name="quotation"]', '.se-toolbar-item-quotation button'],
      divider: ['button[data-name="horizontalLine"]', 'button[data-name="horizontal-line"]', '.se-toolbar-item-horizontalLine button'],
      image: ['button[data-name="image"]', '.se-toolbar-item-image button'],
      fontSize: ['button[data-name="font-size-code"]', '.se-toolbar-option-font-size-code-toolbar-button'],
      bold: ['button[data-name="bold"]', '.se-toolbar-item-bold button'],
      align: ['button[data-name="align"]'],
    },
    imageFileInput: ['input[type="file"]'],
    publishOpen: ['button[class^="publish_btn"]', '.publish_btn__m9KHH', 'button:has-text("발행")'],
    categorySelect: ['button[class^="selectbox_button"]', '.selectbox_button__jb1Dt'],
    categoryOption: (name) => `//span[normalize-space(text())="${name}"]`,
    tagInput: ['#tag-input', 'input[class^="tag_input"]'],
    publishConfirm: ['button[class^="confirm_btn"]', '.confirm_btn__WEaBq', 'button:has-text("발행")'],
  },
};

/** 후보 셀렉터를 순서대로 시도해 처음 잡히는 Locator를 반환한다. */
export async function firstMatch(scope, candidates, { timeout = 4000, visible = true } = {}) {
  const list = Array.isArray(candidates) ? candidates : [candidates];
  for (const sel of list) {
    const loc = scope.locator(sel).first();
    try {
      await loc.waitFor({ state: visible ? 'visible' : 'attached', timeout });
      return loc;
    } catch { /* 다음 후보 */ }
  }
  return null;
}
